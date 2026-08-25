"""Damped least-squares (Levenberg-Marquardt) thickness optimiser."""

from __future__ import annotations

import math
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
        return OptimResult(
            best, best_cost, self.residuals(best), len(history) - 1, True, "coarse", history
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
            return self._optimize_adam(
                layers, free_indices=free_indices, verbose=verbose
            )
        if self.method not in ("lm", "levenberg", "levenberg-marquardt"):
            raise ValueError(
                f"unknown optimizer method {self.method!r}; use 'lm' or 'adam'"
            )
        return self._optimize_lm(layers, free_indices=free_indices, verbose=verbose)

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
        best_x, best_cost, best_r = list(x), cost, r

        m = [0.0] * len(x)
        v = [0.0] * len(x)
        lr = self.adam_lr
        b1, b2, eps = self.adam_beta1, self.adam_beta2, self.adam_eps
        max_step = self.adam_max_step
        stale = 0

        if verbose:
            print(
                f"    Adam start: cost={cost:.6e}  layers={len(materials)}  "
                f"free={len(free)}  lr={lr*1e9:.2f} nm",
                flush=True,
            )

        for it in range(1, self.max_iter + 1):
            g, free_cols = self._cost_gradient(materials, x, cost, free)
            if not free_cols:
                break
            b1t = 1.0 - b1**it
            b2t = 1.0 - b2**it
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
            history.append(cost)
            step_norm = math.sqrt(sum(d * d for d in delta))

            if cost < best_cost - 1e-12:
                best_x, best_cost, best_r = list(x), cost, r
                stale = 0
            else:
                stale += 1
                if stale >= 8:
                    lr = max(lr * 0.5, 0.05e-9)
                    stale = 0

            if verbose and (it == 1 or it % 5 == 0 or it == self.max_iter):
                print(
                    f"    Adam iter {it:3d}: cost={cost:.6e}  "
                    f"best={best_cost:.6e}  lr={lr*1e9:.2f} nm  "
                    f"Σd={sum(x)*1e9:.1f} nm",
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
                )

        return OptimResult(
            list(zip(materials, best_x)),
            best_cost,
            best_r,
            self.max_iter,
            best_cost < history[0],
            "max_iter",
            history,
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
                lam = max(lam * 0.3, 1e-8)
                if verbose and (it == 1 or it % 5 == 0 or it == self.max_iter):
                    total_nm = sum(x) * 1e9
                    print(
                        f"    LM iter {it:3d}: cost={cost:.6e}  "
                        f"λ={lam:.2e}  Σd={total_nm:.1f} nm",
                        flush=True,
                    )
                if step_norm < 1e-12 or cost < self.tol:
                    return OptimResult(
                        list(zip(materials, x)),
                        cost,
                        r,
                        it,
                        True,
                        "converged",
                        history,
                    )
            else:
                lam = min(lam * 3.0, 1e8)

        return OptimResult(
            list(zip(materials, x)),
            cost,
            r,
            self.max_iter,
            cost < history[0],
            "max_iter",
            history,
        )

    def evaluate(self, layers: Sequence[tuple[str, float]]):
        R, T = self._rt(layers)
        ok = specs_satisfied(layers, self.bands, self.wavelengths, R, T)
        return R, T, ok, band_report(self.bands, self.wavelengths, R, T)
