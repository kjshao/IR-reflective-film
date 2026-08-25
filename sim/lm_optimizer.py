"""Damped least-squares (Levenberg-Marquardt) thickness optimiser."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from rt_calculator import RTCalculator

# Default physical bounds for dielectric layers (metres).
DEFAULT_BOUNDS = {
    "tio2": (5e-9, 500e-9),
    "sio2": (5e-9, 550e-9),
    "ito": (15e-9, 150e-9),
    "ag": (6e-9, 25e-9),
    "glass": (50e-9, 500e-9),
    "pet": (50e-9, 500e-9),
    "air": (10e-9, 500e-9),
}


@dataclass
class BandSpec:
    """Optical targets on a closed wavelength interval [wl_lo, wl_hi] (metres)."""

    wl_lo: float
    wl_hi: float
    R_min: float | None = None
    R_max: float | None = None
    T_min: float | None = None
    T_max: float | None = None
    weight: float = 1.0
    R_target: float | None = None
    T_target: float | None = None


@dataclass
class OptimResult:
    layers: list[tuple[str, float]]
    cost: float
    residuals: list[float]
    n_iter: int
    success: bool
    message: str = ""
    history: list[float] = field(default_factory=list)
    # Iteration index (0 = start) at which ``cost`` / ``layers`` were best.
    best_iter: int = 0


def _bounds_for(mat: str) -> tuple[float, float]:
    return DEFAULT_BOUNDS.get(mat.lower(), (10e-9, 500e-9))


def wavelength_grid(bands: Sequence[BandSpec], step: float) -> list[float]:
    """Union of per-band sample grids (inclusive endpoints)."""
    pts: set[float] = set()
    for b in bands:
        n = max(1, int(round((b.wl_hi - b.wl_lo) / step)))
        for i in range(n + 1):
            pts.add(b.wl_lo + i * (b.wl_hi - b.wl_lo) / n)
    return sorted(pts)


def build_epoch_batches(
    n: int,
    batch_size: int,
    sample_stride: int = 1,
    batch_gap: int = 0,
    *,
    start_offset: int = 0,
) -> list[list[int]]:
    """Build all mini-batches of wavelength indices for one epoch.

    Uniform sampling: from ``start_offset``, each batch takes ``batch_size``
    indices spaced by ``sample_stride``. The next batch starts
    ``batch_size * sample_stride + batch_gap`` after the previous start
    (``batch_gap`` is the extra gap between batches).

    Empty trailing batches are dropped. Indices stay in ``[0, n)``.
    """
    if n <= 0:
        return []
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if sample_stride < 1:
        raise ValueError(f"sample_stride must be >= 1, got {sample_stride}")
    if batch_gap < 0:
        raise ValueError(f"batch_gap must be >= 0, got {batch_gap}")
    if start_offset < 0:
        raise ValueError(f"start_offset must be >= 0, got {start_offset}")

    batches: list[list[int]] = []
    i = start_offset
    span = batch_size * sample_stride
    while i < n:
        batch = [i + k * sample_stride for k in range(batch_size) if i + k * sample_stride < n]
        if not batch:
            break
        batches.append(batch)
        i += span + batch_gap
    return batches


# Per-sample pull toward total reflection (R → 1, T → 0).
_DRIVE_SCALE = 0.5
# In-band peak-to-peak (max − min); damps visible ripple vs a wider IR stop.
_RIPPLE_SCALE = 1.5


def _infer_targets(b: BandSpec) -> tuple[float, float]:
    """High-reflector defaults: R_target=1, T_target=0."""
    r_t = 1.0 if b.R_target is None else b.R_target
    t_t = 0.0 if b.T_target is None else b.T_target
    return r_t, t_t


def build_residuals(
    layers: Sequence[tuple[str, float]],
    bands: Sequence[BandSpec],
    wavelengths: Sequence[float],
    R: Sequence[float],
    T: Sequence[float],
    *,
    thickness_weight: float = 0.02,
    thickness_ref: float = 1000e-9,
) -> list[float]:
    """Residuals for broadband total reflection (visible + infrared).

    Every band is a high reflector. Per-sample terms are scaled by
    ``1/sqrt(n)`` so a wider IR grid cannot drown the visible band.
    A peak-to-peak ripple term flattens in-band oscillation.
    """
    ineq: list[float] = []
    drive: list[float] = []
    for b in bands:
        w = math.sqrt(max(b.weight, 0.0))
        r_t, t_t = _infer_targets(b)
        rs, ts = [], []
        for wl, r, t in zip(wavelengths, R, T):
            if b.wl_lo - 1e-15 <= wl <= b.wl_hi + 1e-15:
                rs.append(r)
                ts.append(t)
        if not rs:
            continue
        wn = w / math.sqrt(len(rs))
        for r, t in zip(rs, ts):
            if b.R_min is not None:
                ineq.append(wn * max(0.0, b.R_min - r))
            if b.T_max is not None:
                ineq.append(wn * max(0.0, t - b.T_max))
            drive.append(_DRIVE_SCALE * wn * max(0.0, r_t - r))
            drive.append(_DRIVE_SCALE * wn * max(0.0, t - t_t))
        if b.R_min is not None:
            ineq.append(2.0 * w * max(0.0, b.R_min - min(rs)))
            ineq.append(0.8 * w * max(0.0, (b.R_min + 0.08) - min(rs)))
        if b.T_max is not None:
            ineq.append(1.5 * w * max(0.0, max(ts) - b.T_max))
        drive.append(_RIPPLE_SCALE * w * (max(rs) - min(rs)))
        drive.append(_RIPPLE_SCALE * w * (max(ts) - min(ts)))

    feasible = all(abs(v) <= 1e-12 for v in ineq)
    res = ineq + drive
    if feasible and thickness_weight > 0 and layers:
        total = sum(d for _, d in layers)
        res.append(math.sqrt(thickness_weight) * total / thickness_ref)
    if not res:
        res = [0.0]
    return res


def specs_satisfied(
    layers: Sequence[tuple[str, float]],
    bands: Sequence[BandSpec],
    wavelengths: Sequence[float],
    R: Sequence[float],
    T: Sequence[float],
    tol: float = 1e-4,
) -> bool:
    """True when all band inequalities hold within ``tol``."""
    for b in bands:
        for wl, r, t in zip(wavelengths, R, T):
            if not (b.wl_lo - 1e-15 <= wl <= b.wl_hi + 1e-15):
                continue
            if b.R_min is not None and r < b.R_min - tol:
                return False
            if b.R_max is not None and r > b.R_max + tol:
                return False
            if b.T_min is not None and t < b.T_min - tol:
                return False
            if b.T_max is not None and t > b.T_max + tol:
                return False
    return True


def band_report(
    bands: Sequence[BandSpec],
    wavelengths: Sequence[float],
    R: Sequence[float],
    T: Sequence[float],
) -> list[dict]:
    """Per-band summary statistics for logging."""
    out = []
    for b in bands:
        rs, ts = [], []
        for wl, r, t in zip(wavelengths, R, T):
            if b.wl_lo - 1e-15 <= wl <= b.wl_hi + 1e-15:
                rs.append(r)
                ts.append(t)
        if not rs:
            continue
        out.append(
            {
                "wl_nm": (b.wl_lo * 1e9, b.wl_hi * 1e9),
                "R_mean": sum(rs) / len(rs),
                "R_min": min(rs),
                "R_max": max(rs),
                "T_mean": sum(ts) / len(ts),
                "T_min": min(ts),
                "T_max": max(ts),
                "targets": {
                    "R_min": b.R_min,
                    "R_max": b.R_max,
                    "T_min": b.T_min,
                    "T_max": b.T_max,
                },
            }
        )
    return out


class LMThicknessOptimizer:
    """Thickness optimiser: Levenberg-Marquardt (default) or Adam."""

    def __init__(
        self,
        calculator: RTCalculator,
        bands: Sequence[BandSpec],
        *,
        theta0: float = 0.0,
        incident: str = "air",
        substrate: str = "glass",
        substrate_thickness: float = 0.7e-3,
        exit_medium: str = "air",
        polarization: str = "unpolarized",
        substrate_model: str = "semi_infinite",
        wavelength_step: float = 10e-9,
        thickness_weight: float = 0.02,
        fd_step: float = 0.5e-9,
        lambda0: float = 1e-2,
        max_iter: int = 40,
        tol: float = 1e-8,
        method: str = "lm",
        adam_lr: float = 2e-9,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        adam_max_step: float = 10e-9,
        # Mini-batch Adam (wavelength subsets). Off by default → full-grid Adam.
        mini_batch: bool = False,
        batch_size: int = 8,
        sample_stride: int = 1,
        batch_gap: int = 0,
        n_epochs: int | None = None,
        shuffle_seed: int | None = None,
    ):
        self.calc = calculator
        self.bands = list(bands)
        self.theta0 = theta0
        self.incident = incident
        self.substrate = substrate
        self.substrate_thickness = substrate_thickness
        self.exit_medium = exit_medium
        self.polarization = polarization
        self.substrate_model = substrate_model
        self.wavelengths = wavelength_grid(bands, wavelength_step)
        self.thickness_weight = thickness_weight
        self.fd_step = fd_step
        self.lambda0 = lambda0
        self.max_iter = max_iter
        self.tol = tol
        self.method = (method or "lm").lower()
        self.adam_lr = adam_lr
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps
        self.adam_max_step = adam_max_step
        self.mini_batch = bool(mini_batch)
        self.batch_size = int(batch_size)
        self.sample_stride = int(sample_stride)
        self.batch_gap = int(batch_gap)
        self.n_epochs = max_iter if n_epochs is None else int(n_epochs)
        self.shuffle_seed = shuffle_seed

    def _rt(self, layers: Sequence[tuple[str, float]]):
        return self.calc.spectrum(
            layers,
            self.wavelengths,
            self.theta0,
            incident=self.incident,
            substrate=self.substrate,
            substrate_thickness=self.substrate_thickness,
            exit_medium=self.exit_medium,
            polarization=self.polarization,
            substrate_model=self.substrate_model,
        )

    def residuals(self, layers: Sequence[tuple[str, float]]) -> list[float]:
        R, T = self._rt(layers)
        return build_residuals(
            layers,
            self.bands,
            self.wavelengths,
            R,
            T,
            thickness_weight=self.thickness_weight,
        )

    def cost(self, layers: Sequence[tuple[str, float]]) -> float:
        r = self.residuals(layers)
        return 0.5 * sum(x * x for x in r)

    def _project(self, materials: Sequence[str], x: list[float]) -> list[float]:
        out = []
        for mat, d in zip(materials, x):
            lo, hi = _bounds_for(mat)
            out.append(min(hi, max(lo, d)))
        return out

    def _jacobian(
        self,
        materials: Sequence[str],
        x: list[float],
        r0: list[float],
        free_indices: Sequence[int] | None = None,
    ):
        """Forward-difference Jacobian; only columns in ``free_indices`` are filled."""
        n = len(x)
        m = len(r0)
        free = list(range(n)) if free_indices is None else list(free_indices)
        J = [[0.0] * n for _ in range(m)]
        for j in free:
            step = self.fd_step
            lo, hi = _bounds_for(materials[j])
            xp = list(x)
            if x[j] + step <= hi:
                xp[j] = x[j] + step
                denom = step
            else:
                xp[j] = max(lo, x[j] - step)
                denom = x[j] - xp[j]
                if denom <= 0:
                    continue
            rp = self.residuals(list(zip(materials, xp)))
            sign = 1.0 if xp[j] > x[j] else -1.0
            for i in range(m):
                J[i][j] = sign * (rp[i] - r0[i]) / abs(denom)
        return J, free

    @staticmethod
    def _jtj_jtr(J, r):
        n = len(J[0]) if J else 0
        A = [[0.0] * n for _ in range(n)]
        g = [0.0] * n
        for row, ri in zip(J, r):
            for j in range(n):
                g[j] += row[j] * ri
                for k in range(n):
                    A[j][k] += row[j] * row[k]
        return A, g

    @staticmethod
    def _solve_linear(A, b):
        """Gaussian elimination with partial pivoting; returns None on failure."""
        n = len(b)
        M = [A[i][:] + [b[i]] for i in range(n)]
        for col in range(n):
            piv = max(range(col, n), key=lambda i: abs(M[i][col]))
            if abs(M[piv][col]) < 1e-18:
                return None
            M[col], M[piv] = M[piv], M[col]
            div = M[col][col]
            for k in range(col, n + 1):
                M[col][k] /= div
            for i in range(n):
                if i == col:
                    continue
                factor = M[i][col]
                for k in range(col, n + 1):
                    M[i][k] -= factor * M[col][k]
        return [M[i][n] for i in range(n)]

    def coarse_descent(
        self,
        layers: Sequence[tuple[str, float]],
        *,
        rounds: int = 8,
        step0: float = 8e-9,
        free_indices: Sequence[int] | None = None,
        accept_fn=None,
        verbose: bool = True,
    ) -> OptimResult:
        """Bounded coordinate descent — larger steps than LM finite differences.

        ``accept_fn(layers) -> bool``: optional hard filter (e.g. keep IR stop).
        """
        materials = [m for m, _ in layers]
        x = self._project(materials, [d for _, d in layers])
        free = list(range(len(x))) if free_indices is None else list(free_indices)
        best = list(zip(materials, x))
        if accept_fn is not None and not accept_fn(best):
            # Starting point must be acceptable; otherwise ignore the filter.
            accept_fn = None
        best_cost = self.cost(best)
        history = [best_cost]
        step = step0
        if verbose:
            print(
                f"    coarse start: cost={best_cost:.6e}  step={step*1e9:.1f} nm  "
                f"free={len(free)}/{len(x)}"
                + ("  [constrained]" if accept_fn else ""),
                flush=True,
            )
        for r in range(1, rounds + 1):
            improved = False
            for j in free:
                for delta in (step, -step, 2 * step, -2 * step):
                    trial_x = list(x)
                    trial_x[j] = trial_x[j] + delta
                    trial_x = self._project(materials, trial_x)
                    if abs(trial_x[j] - x[j]) < 1e-15:
                        continue
                    trial = list(zip(materials, trial_x))
                    if accept_fn is not None and not accept_fn(trial):
                        continue
                    c = self.cost(trial)
                    if c < best_cost - 1e-12:
                        best, best_cost, x = trial, c, trial_x
                        improved = True
            history.append(best_cost)
            if verbose and (r == 1 or r % 2 == 0 or not improved):
                print(
                    f"    coarse round {r}: cost={best_cost:.6e}  "
                    f"Σd={sum(x)*1e9:.1f} nm",
                    flush=True,
                )
            if not improved:
                step *= 0.5
                if step < 0.5e-9:
                    break
        best_iter = min(
            range(len(history)),
            key=lambda i: history[i],
        )
        return OptimResult(
            best,
            best_cost,
            self.residuals(best),
            len(history) - 1,
            True,
            "coarse",
            history,
            best_iter=best_iter,
        )

    def _cost_gradient(
        self,
        materials: Sequence[str],
        x: list[float],
        c0: float,
        free_indices: Sequence[int] | None = None,
    ):
        """Forward-difference ∇cost; only ``free_indices`` are filled."""
        n = len(x)
        free = list(range(n)) if free_indices is None else list(free_indices)
        g = [0.0] * n
        for j in free:
            step = self.fd_step
            lo, hi = _bounds_for(materials[j])
            xp = list(x)
            if x[j] + step <= hi:
                xp[j] = x[j] + step
                denom = step
                sign = 1.0
            else:
                xp[j] = max(lo, x[j] - step)
                denom = x[j] - xp[j]
                if denom <= 0:
                    continue
                sign = -1.0
            cp = self.cost(list(zip(materials, xp)))
            g[j] = sign * (cp - c0) / abs(denom)
        return g, free

    def optimize(
        self,
        layers: Sequence[tuple[str, float]],
        *,
        free_indices: Sequence[int] | None = None,
        verbose: bool = True,
    ) -> OptimResult:
        if self.method == "adam":
            if self.mini_batch:
                return self._optimize_adam_minibatch(
                    layers, free_indices=free_indices, verbose=verbose
                )
            return self._optimize_adam(
                layers, free_indices=free_indices, verbose=verbose
            )
        if self.method not in ("lm", "levenberg", "levenberg-marquardt"):
            raise ValueError(
                f"unknown optimizer method {self.method!r}; use 'lm' or 'adam'"
            )
        return self._optimize_lm(layers, free_indices=free_indices, verbose=verbose)

    def _set_wavelengths(self, wavelengths: Sequence[float]) -> None:
        """Swap active wavelength grid (used by mini-batch Adam)."""
        self.wavelengths = list(wavelengths)

    def _adam_step(
        self,
        materials: Sequence[str],
        x: list[float],
        free: Sequence[int],
        m: list[float],
        v: list[float],
        *,
        lr: float,
        t: int,
        cost: float,
    ) -> tuple[list[float], float, list[float], float]:
        """One projected Adam update; returns (x, cost, residuals, step_norm)."""
        b1, b2, eps = self.adam_beta1, self.adam_beta2, self.adam_eps
        max_step = self.adam_max_step
        g, free_cols = self._cost_gradient(materials, x, cost, free)
        if not free_cols:
            r = self.residuals(list(zip(materials, x)))
            return x, cost, r, 0.0
        b1t = 1.0 - b1**t
        b2t = 1.0 - b2**t
        delta = [0.0] * len(x)
        for j in free_cols:
            gj = g[j]
            m[j] = b1 * m[j] + (1.0 - b1) * gj
            v[j] = b2 * v[j] + (1.0 - b2) * gj * gj
            mhat = m[j] / b1t
            vhat = v[j] / b2t
            step = lr * mhat / (math.sqrt(vhat) + eps)
            if step > max_step:
                step = max_step
            elif step < -max_step:
                step = -max_step
            delta[j] = -step

        x = self._project(materials, [x[i] + delta[i] for i in range(len(x))])
        r = self.residuals(list(zip(materials, x)))
        cost = 0.5 * sum(v_i * v_i for v_i in r)
        step_norm = math.sqrt(sum(d * d for d in delta))
        return x, cost, r, step_norm

    def _optimize_adam(
        self,
        layers: Sequence[tuple[str, float]],
        *,
        free_indices: Sequence[int] | None = None,
        verbose: bool = True,
    ) -> OptimResult:
        """Adam on layer thicknesses (projected onto material bounds)."""
        materials = [m for m, _ in layers]
        x = self._project(materials, [d for _, d in layers])
        free = list(range(len(x))) if free_indices is None else list(free_indices)
        r = self.residuals(list(zip(materials, x)))
        cost = 0.5 * sum(v * v for v in r)
        history = [cost]
        best_x, best_cost, best_r, best_iter = list(x), cost, r, 0

        m = [0.0] * len(x)
        v = [0.0] * len(x)
        lr = self.adam_lr
        stale = 0

        if verbose:
            print(
                f"    Adam start: cost={cost:.6e}  layers={len(materials)}  "
                f"free={len(free)}  lr={lr*1e9:.2f} nm",
                flush=True,
            )

        for it in range(1, self.max_iter + 1):
            x, cost, r, step_norm = self._adam_step(
                materials, x, free, m, v, lr=lr, t=it, cost=cost
            )
            history.append(cost)

            if cost < best_cost - 1e-12:
                best_x, best_cost, best_r, best_iter = list(x), cost, r, it
                stale = 0
            else:
                stale += 1
                if stale >= 8:
                    lr = max(lr * 0.5, 0.05e-9)
                    stale = 0

            if verbose and (it == 1 or it % 5 == 0 or it == self.max_iter):
                print(
                    f"    Adam iter {it:3d}: cost={cost:.6e}  "
                    f"best={best_cost:.6e}@iter{best_iter}  "
                    f"lr={lr*1e9:.2f} nm  Σd={sum(best_x)*1e9:.1f} nm",
                    flush=True,
                )
            if step_norm < 1e-12 or best_cost < self.tol:
                return OptimResult(
                    list(zip(materials, best_x)),
                    best_cost,
                    best_r,
                    it,
                    True,
                    "converged",
                    history,
                    best_iter=best_iter,
                )

        return OptimResult(
            list(zip(materials, best_x)),
            best_cost,
            best_r,
            self.max_iter,
            best_cost < history[0],
            "max_iter",
            history,
            best_iter=best_iter,
        )

    def _optimize_adam_minibatch(
        self,
        layers: Sequence[tuple[str, float]],
        *,
        free_indices: Sequence[int] | None = None,
        verbose: bool = True,
    ) -> OptimResult:
        """Adam with per-epoch uniform wavelength mini-batches.

        Each epoch:
          1. Random start offset → build all batches (batch_size, sample_stride,
             batch_gap).
          2. Shuffle batch order and traverse every batch once (one Adam step
             each on that wavelength subset).
          3. Record full-grid cost for history / best tracking.
        """
        full_wls = list(self.wavelengths)
        n_wl = len(full_wls)
        if n_wl == 0:
            raise ValueError("mini-batch Adam needs a non-empty wavelength grid")

        materials = [m for m, _ in layers]
        x = self._project(materials, [d for _, d in layers])
        free = list(range(len(x))) if free_indices is None else list(free_indices)

        self._set_wavelengths(full_wls)
        r = self.residuals(list(zip(materials, x)))
        full_cost = 0.5 * sum(v * v for v in r)
        history = [full_cost]
        best_x, best_cost, best_r, best_iter = list(x), full_cost, r, 0

        m = [0.0] * len(x)
        v = [0.0] * len(x)
        lr = self.adam_lr
        stale = 0
        rng = random.Random(self.shuffle_seed)
        n_epochs = max(1, self.n_epochs)
        # Phase span for random epoch offsets (covers different uniform grids).
        phase_mod = max(1, self.batch_size * self.sample_stride + self.batch_gap)
        t_step = 0

        if verbose:
            print(
                f"    Adam mini-batch start: full_cost={full_cost:.6e}  "
                f"n_wl={n_wl}  batch_size={self.batch_size}  "
                f"sample_stride={self.sample_stride}  batch_gap={self.batch_gap}  "
                f"n_epochs={n_epochs}  lr={lr*1e9:.2f} nm",
                flush=True,
            )

        for epoch in range(1, n_epochs + 1):
            offset = rng.randrange(phase_mod) if phase_mod > 1 else 0
            batches = build_epoch_batches(
                n_wl,
                self.batch_size,
                self.sample_stride,
                self.batch_gap,
                start_offset=offset,
            )
            if not batches:
                # Degenerate config: fall back to full grid for this epoch.
                batches = [list(range(n_wl))]
            order = list(range(len(batches)))
            rng.shuffle(order)

            if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == n_epochs):
                print(
                    f"    epoch {epoch:3d}: offset={offset}  "
                    f"n_batches={len(batches)}  "
                    f"(shuffle → first batch size {len(batches[order[0]])})",
                    flush=True,
                )

            for bi in order:
                idx = batches[bi]
                batch_wls = [full_wls[i] for i in idx]
                self._set_wavelengths(batch_wls)
                batch_layers = list(zip(materials, x))
                batch_r = self.residuals(batch_layers)
                batch_cost = 0.5 * sum(v_i * v_i for v_i in batch_r)
                t_step += 1
                x, _bc, _br, step_norm = self._adam_step(
                    materials, x, free, m, v, lr=lr, t=t_step, cost=batch_cost
                )
                if step_norm < 1e-15:
                    break

            # Full-grid evaluation for comparable history / best selection.
            self._set_wavelengths(full_wls)
            r = self.residuals(list(zip(materials, x)))
            full_cost = 0.5 * sum(v_i * v_i for v_i in r)
            history.append(full_cost)

            if full_cost < best_cost - 1e-12:
                best_x, best_cost, best_r, best_iter = list(x), full_cost, r, epoch
                stale = 0
            else:
                stale += 1
                if stale >= 8:
                    lr = max(lr * 0.5, 0.05e-9)
                    stale = 0

            if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == n_epochs):
                print(
                    f"    Adam epoch {epoch:3d}: full_cost={full_cost:.6e}  "
                    f"best={best_cost:.6e}@epoch{best_iter}  "
                    f"lr={lr*1e9:.2f} nm  Σd={sum(best_x)*1e9:.1f} nm",
                    flush=True,
                )
            if best_cost < self.tol:
                self._set_wavelengths(full_wls)
                return OptimResult(
                    list(zip(materials, best_x)),
                    best_cost,
                    best_r,
                    epoch,
                    True,
                    "converged",
                    history,
                    best_iter=best_iter,
                )

        self._set_wavelengths(full_wls)
        return OptimResult(
            list(zip(materials, best_x)),
            best_cost,
            best_r,
            n_epochs,
            best_cost < history[0],
            "max_epochs",
            history,
            best_iter=best_iter,
        )

    def _optimize_lm(
        self,
        layers: Sequence[tuple[str, float]],
        *,
        free_indices: Sequence[int] | None = None,
        verbose: bool = True,
    ) -> OptimResult:
        materials = [m for m, _ in layers]
        x = self._project(materials, [d for _, d in layers])
        free = list(range(len(x))) if free_indices is None else list(free_indices)
        lam = self.lambda0
        history: list[float] = []
        r = self.residuals(list(zip(materials, x)))
        cost = 0.5 * sum(v * v for v in r)
        history.append(cost)
        best_x, best_cost, best_r, best_iter = list(x), cost, r, 0

        if verbose:
            print(
                f"    LM start: cost={cost:.6e}  layers={len(materials)}  "
                f"free={len(free)}",
                flush=True,
            )

        for it in range(1, self.max_iter + 1):
            J, free_cols = self._jacobian(materials, x, r, free)
            # Reduce to free subspace for the linear solve.
            nf = len(free_cols)
            if nf == 0:
                break
            Jf = [[row[j] for j in free_cols] for row in J]
            A, g = self._jtj_jtr(Jf, r)

            Ad = [row[:] for row in A]
            for i in range(nf):
                Ad[i][i] += lam * (A[i][i] + 1e-12)
            rhs = [-gi for gi in g]
            delta_f = self._solve_linear(Ad, rhs)
            if delta_f is None:
                lam = min(lam * 10.0, 1e8)
                continue

            delta = [0.0] * len(x)
            for k, j in enumerate(free_cols):
                delta[j] = delta_f[k]

            x_trial = self._project(materials, [x[i] + delta[i] for i in range(len(x))])
            r_trial = self.residuals(list(zip(materials, x_trial)))
            cost_trial = 0.5 * sum(v * v for v in r_trial)

            if cost_trial < cost * (1.0 - 1e-12):
                step_norm = math.sqrt(sum(d * d for d in delta))
                x, r, cost = x_trial, r_trial, cost_trial
                history.append(cost)
                if cost < best_cost - 1e-12:
                    best_x, best_cost, best_r, best_iter = list(x), cost, r, it
                lam = max(lam * 0.3, 1e-8)
                if verbose and (it == 1 or it % 5 == 0 or it == self.max_iter):
                    total_nm = sum(best_x) * 1e9
                    print(
                        f"    LM iter {it:3d}: cost={cost:.6e}  "
                        f"best={best_cost:.6e}@iter{best_iter}  "
                        f"λ={lam:.2e}  Σd={total_nm:.1f} nm",
                        flush=True,
                    )
                if step_norm < 1e-12 or best_cost < self.tol:
                    return OptimResult(
                        list(zip(materials, best_x)),
                        best_cost,
                        best_r,
                        it,
                        True,
                        "converged",
                        history,
                        best_iter=best_iter,
                    )
            else:
                lam = min(lam * 3.0, 1e8)

        return OptimResult(
            list(zip(materials, best_x)),
            best_cost,
            best_r,
            self.max_iter,
            best_cost < history[0],
            "max_iter",
            history,
            best_iter=best_iter,
        )

    def evaluate(self, layers: Sequence[tuple[str, float]]):
        R, T = self._rt(layers)
        ok = specs_satisfied(layers, self.bands, self.wavelengths, R, T)
        return R, T, ok, band_report(self.bands, self.wavelengths, R, T)
