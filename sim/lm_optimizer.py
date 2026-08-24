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


def _infer_targets(b: BandSpec) -> tuple[float | None, float | None]:
    r_t, t_t = b.R_target, b.T_target
    if r_t is None:
        if b.R_min is not None and b.R_max is None:
            r_t = min(1.0, b.R_min + 0.20)
        elif b.R_max is not None and b.R_min is None:
            r_t = 0.0
        elif b.R_min is not None and b.R_max is not None:
            r_t = 0.5 * (b.R_min + b.R_max)
    if t_t is None:
        if b.T_min is not None and b.T_max is None:
            t_t = min(1.0, b.T_min + 0.02)
        elif b.T_max is not None and b.T_min is None:
            t_t = 0.0
        elif b.T_min is not None and b.T_max is not None:
            t_t = 0.5 * (b.T_min + b.T_max)
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
    """Inequality residuals always; thickness/smooth only when feasible.

    Soft inequalities stay in the residual vector even when currently zero so
    that a subsequent thickness-minimisation step cannot quietly violate them
    without a restoring gradient.
    """
    ineq: list[float] = []
    smooth: list[float] = []
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
        r_mean = sum(rs) / len(rs)
        t_mean = sum(ts) / len(ts)
        if b.R_min is not None:
            ineq.append(w * max(0.0, b.R_min - min(rs)))
        if b.R_max is not None:
            ineq.append(w * max(0.0, max(rs) - b.R_max))
        if b.T_min is not None:
            ineq.append(w * max(0.0, b.T_min - min(ts)))
        if b.T_max is not None:
            ineq.append(w * max(0.0, max(ts) - b.T_max))
        if r_t is not None:
            smooth.append(0.35 * w * (r_mean - r_t))
        if t_t is not None:
            smooth.append(0.35 * w * (t_mean - t_t))

    feasible = all(abs(v) <= 1e-12 for v in ineq)
    res = list(ineq)
    if feasible:
        res.extend(smooth)
        if thickness_weight > 0 and layers:
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
    """Levenberg-Marquardt optimiser over layer thicknesses only."""

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

    def _jacobian(self, materials: Sequence[str], x: list[float], r0: list[float]):
        """Forward-difference Jacobian, shape (m, n)."""
        n = len(x)
        m = len(r0)
        J = [[0.0] * n for _ in range(m)]
        for j in range(n):
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
        return J

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
        verbose: bool = True,
    ) -> OptimResult:
        """Bounded coordinate descent — larger steps than LM finite differences."""
        materials = [m for m, _ in layers]
        x = self._project(materials, [d for _, d in layers])
        best = list(zip(materials, x))
        best_cost = self.cost(best)
        history = [best_cost]
        step = step0
        if verbose:
            print(
                f"    coarse start: cost={best_cost:.6e}  step={step*1e9:.1f} nm",
                flush=True,
            )
        for r in range(1, rounds + 1):
            improved = False
            for j in range(len(x)):
                for delta in (step, -step):
                    trial_x = list(x)
                    trial_x[j] = trial_x[j] + delta
                    trial_x = self._project(materials, trial_x)
                    if abs(trial_x[j] - x[j]) < 1e-15:
                        continue
                    trial = list(zip(materials, trial_x))
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

    def optimize(
        self,
        layers: Sequence[tuple[str, float]],
        *,
        verbose: bool = True,
    ) -> OptimResult:
        materials = [m for m, _ in layers]
        x = self._project(materials, [d for _, d in layers])
        lam = self.lambda0
        history: list[float] = []
        r = self.residuals(list(zip(materials, x)))
        cost = 0.5 * sum(v * v for v in r)
        history.append(cost)

        if verbose:
            print(f"    LM start: cost={cost:.6e}  layers={len(materials)}")

        for it in range(1, self.max_iter + 1):
            J = self._jacobian(materials, x, r)
            A, g = self._jtj_jtr(J, r)
            n = len(x)
            if n == 0:
                break

            # Damping on diagonal (Levenberg).
            Ad = [row[:] for row in A]
            for i in range(n):
                Ad[i][i] += lam * (A[i][i] + 1e-12)
            # Solve (J^T J + λ diag) δ = -J^T r
            rhs = [-gi for gi in g]
            delta = self._solve_linear(Ad, rhs)
            if delta is None:
                lam = min(lam * 10.0, 1e8)
                continue

            x_trial = self._project(materials, [x[i] + delta[i] for i in range(n)])
            r_trial = self.residuals(list(zip(materials, x_trial)))
            cost_trial = 0.5 * sum(v * v for v in r_trial)

            # Actual vs predicted reduction (Marquardt gain ratio).
            pred = 0.0
            for i in range(n):
                pred += -g[i] * delta[i]
                for k in range(n):
                    pred += 0.5 * delta[i] * A[i][k] * delta[k]
            # Prefer simple accept/reject on cost for soft inequalities.
            if cost_trial < cost * (1.0 - 1e-12) or (
                abs(cost_trial - cost) < self.tol and pred > 0
            ):
                step_norm = math.sqrt(sum(d * d for d in delta))
                x, r, cost = x_trial, r_trial, cost_trial
                history.append(cost)
                lam = max(lam * 0.3, 1e-8)
                if verbose and (it == 1 or it % 5 == 0 or it == self.max_iter):
                    total_nm = sum(x) * 1e9
                    print(
                        f"    LM iter {it:3d}: cost={cost:.6e}  "
                        f"λ={lam:.2e}  Σd={total_nm:.1f} nm"
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
