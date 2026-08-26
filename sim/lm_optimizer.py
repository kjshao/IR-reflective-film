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


def build_wrapped_wavelength_batch(
    wl_lo: float,
    wl_hi: float,
    batch_size: int,
    start: float,
) -> list[float]:
    """Uniform wavelengths from ``start``, wrapping at ``wl_hi`` back to ``wl_lo``.

    Spacing is ``(wl_hi - wl_lo) / batch_size`` so the ``batch_size`` points
    cover the study interval evenly. ``start`` may be any float; it is first
    mapped into ``[wl_lo, wl_hi)`` before stepping.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    span = wl_hi - wl_lo
    if span <= 0.0:
        raise ValueError(f"need wl_hi > wl_lo, got [{wl_lo}, {wl_hi}]")
    # Fold start into [wl_lo, wl_hi).
    s0 = wl_lo + ((start - wl_lo) % span)
    delta = span / batch_size
    return [wl_lo + ((s0 - wl_lo + k * delta) % span) for k in range(batch_size)]


def build_epoch_wavelength_batches(
    wl_lo: float,
    wl_hi: float,
    batch_size: int,
    n_batches: int,
    rng: random.Random,
) -> list[tuple[float, list[float]]]:
    """Build ``n_batches`` wrapped uniform wavelength batches for one epoch.

    Each batch draws an independent float start uniformly in ``[wl_lo, wl_hi)``.
    Returns a list of ``(start, wavelengths)``.
    """
    if n_batches < 1:
        raise ValueError(f"n_batches must be >= 1, got {n_batches}")
    span = wl_hi - wl_lo
    if span <= 0.0:
        raise ValueError(f"need wl_hi > wl_lo, got [{wl_lo}, {wl_hi}]")
    out: list[tuple[float, list[float]]] = []
    for _ in range(n_batches):
        start = wl_lo + rng.random() * span
        out.append((start, build_wrapped_wavelength_batch(wl_lo, wl_hi, batch_size, start)))
    return out


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


# Global (scipy) method aliases → canonical name.
_GLOBAL_METHODS = {
    "de": "de",
    "differential_evolution": "de",
    "diffevo": "de",
    "da": "da",
    "dual_annealing": "da",
    "annealing": "da",
}


def _require_scipy():
    try:
        import scipy.optimize as spo
    except ImportError as exc:
        raise SystemExit(
            "scipy is required for global methods "
            "('de' / 'dual_annealing'); pip install scipy"
        ) from exc
    return spo


class LMThicknessOptimizer:
    """Thickness optimiser: LM, Adam, or scipy global (DE / dual annealing)."""

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
        n_batches: int | None = None,
        n_epochs: int | None = None,
        shuffle_seed: int | None = None,
        # scipy global search (differential_evolution / dual_annealing)
        de_popsize: int = 15,
        de_mutation: float | tuple[float, float] = (0.5, 1.0),
        de_recombination: float = 0.7,
        global_seed: int | None = None,
        global_polish: bool = True,
        global_polish_lm: bool = False,
        da_initial_temp: float = 5230.0,
        da_visit: float = 2.62,
        da_accept: float = -5.0,
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
        # Default: about one full pass over the discrete study grid per epoch.
        if n_batches is None:
            n_wl = max(1, len(self.wavelengths))
            self.n_batches = max(1, n_wl // max(1, self.batch_size))
        else:
            self.n_batches = int(n_batches)
        self.n_epochs = max_iter if n_epochs is None else int(n_epochs)
        self.shuffle_seed = shuffle_seed
        self.de_popsize = int(de_popsize)
        self.de_mutation = de_mutation
        self.de_recombination = float(de_recombination)
        self.global_seed = global_seed
        self.global_polish = bool(global_polish)
        self.global_polish_lm = bool(global_polish_lm)
        self.da_initial_temp = float(da_initial_temp)
        self.da_visit = float(da_visit)
        self.da_accept = float(da_accept)

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
        global_name = _GLOBAL_METHODS.get(self.method)
        if global_name == "de":
            return self._optimize_differential_evolution(
                layers, free_indices=free_indices, verbose=verbose
            )
        if global_name == "da":
            return self._optimize_dual_annealing(
                layers, free_indices=free_indices, verbose=verbose
            )
        if self.method not in ("lm", "levenberg", "levenberg-marquardt"):
            raise ValueError(
                f"unknown optimizer method {self.method!r}; "
                "use 'lm', 'adam', 'de', or 'dual_annealing'"
            )
        return self._optimize_lm(layers, free_indices=free_indices, verbose=verbose)

    def _global_objective_setup(
        self,
        layers: Sequence[tuple[str, float]],
        free_indices: Sequence[int] | None,
        *,
        verbose: bool = False,
        tag: str = "global",
    ):
        """Shared scaffolding for scipy global methods on free thicknesses."""
        materials = [m for m, _ in layers]
        x0_full = self._project(materials, [d for _, d in layers])
        free = list(range(len(x0_full))) if free_indices is None else list(free_indices)
        if not free:
            raise ValueError("global optimisation needs at least one free layer")
        bounds = [_bounds_for(materials[j]) for j in free]
        x0_free = [x0_full[j] for j in free]
        start_layers = list(zip(materials, x0_full))
        start_cost = self.cost(start_layers)
        history = [start_cost]
        best = {
            "x": list(x0_full),
            "cost": start_cost,
            "n_eval": 0,
            "n_improve": 0,
        }
        progress = {"last_heartbeat": 0}

        def _sum_nm(x_vals: Sequence[float]) -> float:
            return sum(x_vals) * 1e9

        def objective(x_free) -> float:
            x = list(x0_full)
            for k, j in enumerate(free):
                x[j] = float(x_free[k])
            x = self._project(materials, x)
            c = self.cost(list(zip(materials, x)))
            best["n_eval"] += 1
            if c < best["cost"] - 1e-15:
                best["cost"] = c
                best["x"] = list(x)
                best["n_improve"] += 1
                history.append(c)
                if verbose:
                    print(
                        f"    {tag} eval {best['n_eval']:5d}: "
                        f"NEW best={c:.6e}  "
                        f"Δ={c - start_cost:+.3e}  "
                        f"Σd={_sum_nm(x):.1f} nm  "
                        f"improves={best['n_improve']}",
                        flush=True,
                    )
            elif verbose and best["n_eval"] - progress["last_heartbeat"] >= 50:
                progress["last_heartbeat"] = best["n_eval"]
                print(
                    f"    {tag} eval {best['n_eval']:5d}: "
                    f"cost={c:.6e}  best={best['cost']:.6e}  "
                    f"Σd_best={_sum_nm(best['x']):.1f} nm",
                    flush=True,
                )
            return c

        return (
            materials,
            x0_full,
            free,
            bounds,
            x0_free,
            start_cost,
            history,
            best,
            objective,
            _sum_nm,
        )

    def _maybe_polish_lm(
        self,
        layers: Sequence[tuple[str, float]],
        *,
        free_indices: Sequence[int] | None,
        history: list[float],
        verbose: bool,
        label: str,
    ) -> OptimResult:
        if not self.global_polish_lm:
            cost = self.cost(layers)
            best_iter = (
                min(range(len(history)), key=lambda i: history[i]) if history else 0
            )
            return OptimResult(
                list(layers),
                cost,
                self.residuals(layers),
                max(0, len(history) - 1),
                True,
                label,
                history,
                best_iter=best_iter,
            )
        if verbose:
            print(f"    {label}: LM polish …", flush=True)
        polished = self._optimize_lm(
            layers, free_indices=free_indices, verbose=verbose
        )
        merged = list(history) + list(polished.history[1:])
        best_iter = min(range(len(merged)), key=lambda i: merged[i])
        return OptimResult(
            polished.layers,
            polished.cost,
            polished.residuals,
            polished.n_iter + max(0, len(history) - 1),
            polished.success,
            f"{label}+lm",
            merged,
            best_iter=best_iter,
        )

    def _optimize_differential_evolution(
        self,
        layers: Sequence[tuple[str, float]],
        *,
        free_indices: Sequence[int] | None = None,
        verbose: bool = True,
    ) -> OptimResult:
        """Bounded differential evolution (scipy) over free thicknesses."""
        spo = _require_scipy()
        (
            materials,
            _x0_full,
            free,
            bounds,
            x0_free,
            start_cost,
            history,
            best,
            objective,
            sum_nm,
        ) = self._global_objective_setup(
            layers, free_indices, verbose=verbose, tag="DE"
        )

        n_pop = max(5, self.de_popsize * len(free))
        if verbose:
            print(
                f"    DE start: cost={start_cost:.6e}  layers={len(materials)}  "
                f"free={len(free)}  popsize={self.de_popsize} "
                f"(~{n_pop} individuals)  maxiter={self.max_iter}  "
                f"mutation={self.de_mutation}  recom={self.de_recombination}  "
                f"polish={self.global_polish}  seed={self.global_seed}",
                flush=True,
            )
            print(
                f"    DE start: Σd={sum_nm(_x0_full):.1f} nm  "
                f"free thicknesses (nm) = "
                f"{[round(x0_free[i] * 1e9, 2) for i in range(len(free))]}",
                flush=True,
            )

        last_logged_gen = {"g": -1}

        def callback(intermediate_result, convergence=None):
            # scipy≥1.15 passes OptimizeResult; older passes (xk, convergence).
            if hasattr(intermediate_result, "fun"):
                fun = float(intermediate_result.fun)
                gen = int(getattr(intermediate_result, "nit", 0))
                conv = getattr(intermediate_result, "convergence", convergence)
            else:
                fun = best["cost"]
                gen = last_logged_gen["g"] + 1
                conv = convergence
            if gen == last_logged_gen["g"]:
                return False
            last_logged_gen["g"] = gen
            if not verbose:
                return False
            conv_s = f"{float(conv):.3e}" if conv is not None else "n/a"
            print(
                f"    DE gen {gen:3d}/{self.max_iter}: "
                f"pop_best={fun:.6e}  best={best['cost']:.6e}  "
                f"Δ={best['cost'] - start_cost:+.3e}  "
                f"evals={best['n_eval']}  improves={best['n_improve']}  "
                f"Σd={sum_nm(best['x']):.1f} nm  conv={conv_s}",
                flush=True,
            )
            return False

        result = spo.differential_evolution(
            objective,
            bounds,
            maxiter=self.max_iter,
            popsize=self.de_popsize,
            mutation=self.de_mutation,
            recombination=self.de_recombination,
            seed=self.global_seed,
            polish=self.global_polish,
            init="latinhypercube",
            x0=x0_free,
            atol=self.tol,
            tol=0.01,
            workers=1,
            updating="immediate",
            callback=callback,
            disp=False,
        )
        x = list(_x0_full)
        for k, j in enumerate(free):
            x[j] = float(result.x[k])
        x = self._project(materials, x)
        result_cost = self.cost(list(zip(materials, x)))
        # Prefer the best point seen during search (polish can move off it).
        if best["cost"] <= result_cost + 1e-15:
            x = list(best["x"])
        layers_best = list(zip(materials, x))
        final_cost = self.cost(layers_best)
        if abs(final_cost - history[-1]) > 1e-15:
            history.append(final_cost)
        if verbose:
            free_nm = [round(x[j] * 1e9, 2) for j in free]
            print(
                f"    DE done: cost={final_cost:.6e}  "
                f"Δ={final_cost - start_cost:+.3e}  "
                f"success={bool(result.success)}  "
                f"evals={best['n_eval']}  improves={best['n_improve']}  "
                f"Σd={sum_nm(x):.1f} nm",
                flush=True,
            )
            print(
                f"    DE done: free thicknesses (nm) = {free_nm}  "
                f"msg={result.message}",
                flush=True,
            )
        return self._maybe_polish_lm(
            layers_best,
            free_indices=free_indices,
            history=history,
            verbose=verbose,
            label="de",
        )

    def _optimize_dual_annealing(
        self,
        layers: Sequence[tuple[str, float]],
        *,
        free_indices: Sequence[int] | None = None,
        verbose: bool = True,
    ) -> OptimResult:
        """Dual annealing (scipy) over free thicknesses."""
        spo = _require_scipy()
        (
            materials,
            _x0_full,
            free,
            bounds,
            x0_free,
            start_cost,
            history,
            best,
            objective,
            sum_nm,
        ) = self._global_objective_setup(
            layers, free_indices, verbose=verbose, tag="DA"
        )

        context_name = {0: "accept", 1: "local", 2: "step_done"}
        if verbose:
            print(
                f"    DA start: cost={start_cost:.6e}  layers={len(materials)}  "
                f"free={len(free)}  maxiter={self.max_iter}  "
                f"T0={self.da_initial_temp:g}  visit={self.da_visit:g}  "
                f"accept={self.da_accept:g}  "
                f"local_search={self.global_polish}  seed={self.global_seed}",
                flush=True,
            )
            print(
                f"    DA start: Σd={sum_nm(_x0_full):.1f} nm  "
                f"free thicknesses (nm) = "
                f"{[round(x0_free[i] * 1e9, 2) for i in range(len(free))]}",
                flush=True,
            )

        step_count = {"n": 0}

        def callback(x_free, f, context):
            # context: 0=acceptance, 1=local search, 2=strategy step finished
            step_count["n"] += 1
            if not verbose:
                return False
            name = context_name.get(int(context), str(context))
            # Log every accept/local and every few finished steps.
            if int(context) == 2 and step_count["n"] % 3 != 0:
                return False
            x_tmp = list(_x0_full)
            for k, j in enumerate(free):
                x_tmp[j] = float(x_free[k])
            x_tmp = self._project(materials, x_tmp)
            print(
                f"    DA step {step_count['n']:4d} [{name:9s}]: "
                f"f={float(f):.6e}  best={best['cost']:.6e}  "
                f"Δ={best['cost'] - start_cost:+.3e}  "
                f"evals={best['n_eval']}  improves={best['n_improve']}  "
                f"Σd={sum_nm(best['x']):.1f} nm  "
                f"Σd_cur={sum_nm(x_tmp):.1f} nm",
                flush=True,
            )
            return False  # continue

        kw = dict(
            func=objective,
            bounds=bounds,
            maxiter=self.max_iter,
            initial_temp=self.da_initial_temp,
            visit=self.da_visit,
            accept=self.da_accept,
            seed=self.global_seed,
            x0=x0_free,
            callback=callback,
            no_local_search=not self.global_polish,
        )
        result = spo.dual_annealing(**kw)
        x = list(_x0_full)
        for k, j in enumerate(free):
            x[j] = float(result.x[k])
        x = self._project(materials, x)
        result_cost = self.cost(list(zip(materials, x)))
        if best["cost"] <= result_cost + 1e-15:
            x = list(best["x"])
        layers_best = list(zip(materials, x))
        final_cost = self.cost(layers_best)
        if abs(final_cost - history[-1]) > 1e-15:
            history.append(final_cost)
        if verbose:
            free_nm = [round(x[j] * 1e9, 2) for j in free]
            print(
                f"    DA done: cost={final_cost:.6e}  "
                f"Δ={final_cost - start_cost:+.3e}  "
                f"success={bool(result.success)}  "
                f"evals={best['n_eval']}  improves={best['n_improve']}  "
                f"Σd={sum_nm(x):.1f} nm  steps={step_count['n']}",
                flush=True,
            )
            print(
                f"    DA done: free thicknesses (nm) = {free_nm}  "
                f"msg={result.message}",
                flush=True,
            )
        return self._maybe_polish_lm(
            layers_best,
            free_indices=free_indices,
            history=history,
            verbose=verbose,
            label="dual_annealing",
        )

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
        """Adam with per-epoch wrapped uniform wavelength mini-batches.

        Each epoch:
          1. Draw ``n_batches`` independent float starts in ``[wl_min, wl_max)``.
          2. Each batch takes ``batch_size`` points spaced by
             ``(wl_max-wl_min)/batch_size``, wrapping at ``wl_max`` back to
             ``wl_min``.
          3. One Adam step per batch; then record full-grid cost for history /
             best tracking.
        """
        full_wls = list(self.wavelengths)
        n_wl = len(full_wls)
        if n_wl == 0:
            raise ValueError("mini-batch Adam needs a non-empty wavelength grid")
        wl_lo = full_wls[0]
        wl_hi = full_wls[-1]
        if wl_hi <= wl_lo:
            raise ValueError(
                f"mini-batch needs wl_max > wl_min, got [{wl_lo}, {wl_hi}]"
            )

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
        n_batches = max(1, self.n_batches)
        batch_size = max(1, self.batch_size)
        t_step = 0
        nm = 1e9

        if verbose:
            print(
                f"    Adam mini-batch start: full_cost={full_cost:.6e}  "
                f"n_wl={n_wl}  wl=[{wl_lo*nm:.2f}, {wl_hi*nm:.2f}] nm  "
                f"batch_size={batch_size}  n_batches={n_batches}  "
                f"n_epochs={n_epochs}  lr={lr*nm:.2f} nm",
                flush=True,
            )

        for epoch in range(1, n_epochs + 1):
            batches = build_epoch_wavelength_batches(
                wl_lo, wl_hi, batch_size, n_batches, rng
            )
            order = list(range(len(batches)))
            rng.shuffle(order)

            if verbose:
                print(
                    f"    epoch {epoch:3d}/{n_epochs}: "
                    f"n_batches={n_batches}  batch_size={batch_size}",
                    flush=True,
                )

            epoch_batch_cost_sum = 0.0
            for bi_pos, bi in enumerate(order, 1):
                start, batch_wls = batches[bi]
                self._set_wavelengths(batch_wls)
                batch_layers = list(zip(materials, x))
                batch_r = self.residuals(batch_layers)
                batch_cost = 0.5 * sum(v_i * v_i for v_i in batch_r)
                t_step += 1
                x, _bc, _br, step_norm = self._adam_step(
                    materials, x, free, m, v, lr=lr, t=t_step, cost=batch_cost
                )
                epoch_batch_cost_sum += batch_cost
                if verbose:
                    wl_sorted = sorted(batch_wls)
                    print(
                        f"      batch {bi_pos:3d}/{n_batches}: "
                        f"start={start*nm:.4f} nm  "
                        f"wl=[{wl_sorted[0]*nm:.2f}, {wl_sorted[-1]*nm:.2f}] nm  "
                        f"cost={batch_cost:.6e}  "
                        f"|Δ|={step_norm*nm:.4f} nm",
                        flush=True,
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

            if verbose:
                mean_batch = epoch_batch_cost_sum / max(1, n_batches)
                print(
                    f"    epoch {epoch:3d} done: full_cost={full_cost:.6e}  "
                    f"mean_batch_cost={mean_batch:.6e}  "
                    f"best={best_cost:.6e}@epoch{best_iter}  "
                    f"lr={lr*nm:.2f} nm  Σd={sum(best_x)*nm:.1f} nm",
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
