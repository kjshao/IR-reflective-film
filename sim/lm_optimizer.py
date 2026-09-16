"""Damped least-squares (Levenberg-Marquardt) thickness optimiser."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from rt_calculator import RTCalculator

# Default physical bounds for dielectric layers (metres).
DEFAULT_BOUNDS = {
    "tio2": (5e-9, 500e-9),
    "tio2_pvd": (5e-9, 500e-9),
    "tio2_sputter": (5e-9, 500e-9),
    "tio2_eb": (5e-9, 500e-9),
    "tio2_amorphous": (5e-9, 500e-9),
    "tio2_a": (5e-9, 500e-9),
    "tio2_rutile": (5e-9, 500e-9),
    "sio2": (5e-9, 550e-9),
    "sio2_pvd": (5e-9, 550e-9),
    "sio2_fused": (5e-9, 550e-9),
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


def thickness_rms_nm(x: Sequence[float], x0: Sequence[float]) -> float:
    """RMS thickness change from ``x0`` to ``x``, in nanometres."""
    n = min(len(x), len(x0))
    if n <= 0:
        return 0.0
    acc = 0.0
    for i in range(n):
        d = (float(x[i]) - float(x0[i])) * 1e9
        acc += d * d
    return math.sqrt(acc / n)


def is_better_checkpoint(
    cand_cost: float,
    best_cost: float,
) -> bool:
    """Whether ``cand`` has a strictly lower loss than the running best."""
    return float(cand_cost) < float(best_cost) - 1e-15


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
    # For DE/DA + local polish: the global-stage result before LM/Adam.
    pre_polish: OptimResult | None = None
    # Last iterate (may differ from best ``layers`` / ``cost``).
    final_layers: list[tuple[str, float]] | None = None
    final_cost: float | None = None
    start_cost: float | None = None


def resolve_global_polish_method(
    method: str | None = None,
    *,
    global_polish_lm: bool | None = None,
) -> str:
    """Return the canonical local method used after a global search."""
    if method is not None and str(method).strip() != "":
        m = str(method).lower().strip()
        aliases = {
            "none": "none",
            "off": "none",
            "false": "none",
            "0": "none",
            "lm": "lm",
            "levenberg": "lm",
            "levenberg-marquardt": "lm",
            "adam": "adam",
            "cg": "cg",
            "conjugate_gradient": "cg",
            "conjugate-gradient": "cg",
            "ncg": "cg",
            "lbfgs": "lbfgs",
            "l-bfgs": "lbfgs",
            "lbfgsb": "lbfgs",
            "l-bfgs-b": "lbfgs",
            "trf": "trf",
            "least_squares": "trf",
            "least-squares": "trf",
        }
        if m not in aliases:
            raise ValueError(
                f"unknown global_polish_method {method!r}; "
                "use none|lm|adam|cg|lbfgs|trf"
            )
        return aliases[m]
    if global_polish_lm:
        return "lm"
    return "none"


def parse_thickness_bounds_nm(value) -> dict[str, tuple[float, float]]:
    """Parse JSON per-material thickness bounds and convert nm to metres."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("thickness_bounds_nm must be a JSON object")
    parsed: dict[str, tuple[float, float]] = {}
    for material, bounds in value.items():
        if (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or isinstance(bounds[0], bool)
            or isinstance(bounds[1], bool)
        ):
            raise ValueError(
                f"thickness_bounds_nm.{material} must be [min_nm, max_nm]"
            )
        try:
            lo_nm, hi_nm = float(bounds[0]), float(bounds[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"thickness_bounds_nm.{material} must contain numbers"
            ) from exc
        if not math.isfinite(lo_nm) or not math.isfinite(hi_nm):
            raise ValueError(
                f"thickness_bounds_nm.{material} must contain finite numbers"
            )
        if lo_nm < 0.0 or hi_nm <= lo_nm:
            raise ValueError(
                f"thickness_bounds_nm.{material} requires 0 <= min_nm < max_nm"
            )
        key = str(material).lower().replace("-", "_")
        parsed[key] = (lo_nm * 1e-9, hi_nm * 1e-9)
    return parsed


def _bounds_for(
    mat: str,
    min_thickness: float | None = None,
    thickness_bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[float, float]:
    """Return (lo, hi) thickness bounds in metres for ``mat``.

    ``min_thickness`` (metres) raises the lower bound for every material:
    ``lo = max(material_lo, min_thickness)``. If that would exceed ``hi``,
    ``lo`` is clamped to ``hi``.
    """
    key = mat.lower().replace("-", "_")
    configured = thickness_bounds or {}
    lo, hi = configured.get(key, DEFAULT_BOUNDS.get(key, (10e-9, 500e-9)))
    if min_thickness is not None:
        lo = max(lo, float(min_thickness))
        if lo > hi:
            lo = hi
    return lo, hi


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
    """Return a fixed-length, band-normalized residual vector.

    All four inequality directions are supported.  Targets are optional and
    never inferred: a pass band must not silently become a high-reflector
    objective.  Keeping the vector length independent of feasibility is
    required by Gauss--Newton/TRF Jacobians.
    """
    residuals: list[float] = []
    active = []
    for b in bands:
        samples = [
            (r, t)
            for wl, r, t in zip(wavelengths, R, T)
            if b.wl_lo - 1e-15 <= wl <= b.wl_hi + 1e-15
        ]
        if samples and b.weight > 0.0:
            active.append((b, samples))

    weight_sum = sum(float(b.weight) for b, _ in active)
    for b, samples in active:
        if weight_sum <= 0.0:
            continue
        # 0.5 * sum(residual**2) is a weighted mean-square merit.
        scale = math.sqrt(2.0 * float(b.weight) / (weight_sum * len(samples)))
        for r, t in samples:
            if b.R_min is not None:
                residuals.append(scale * max(0.0, b.R_min - r))
            if b.R_max is not None:
                residuals.append(scale * max(0.0, r - b.R_max))
            if b.T_min is not None:
                residuals.append(scale * max(0.0, b.T_min - t))
            if b.T_max is not None:
                residuals.append(scale * max(0.0, t - b.T_max))
            if b.R_target is not None:
                residuals.append(scale * (r - b.R_target))
            if b.T_target is not None:
                residuals.append(scale * (t - b.T_target))

    if thickness_weight > 0 and layers:
        total = sum(d for _, d in layers)
        residuals.append(
            math.sqrt(2.0 * thickness_weight) * total / thickness_ref
        )
    return residuals or [0.0]


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
            "scipy is required for 'auto', 'multistart', 'trf', 'lbfgs', "
            "'de', and 'dual_annealing'; "
            "pip install scipy"
        ) from exc
    return spo


class LMThicknessOptimizer:
    """Thickness optimiser: LM, Adam, CG, L-BFGS-B, or scipy global methods."""

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
        method: str = "trf",
        adam_lr: float = 2e-9,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        adam_max_step: float = 10e-9,
        # Nonlinear CG (Polak–Ribière) on scalar cost; defaults track Adam.
        cg_initial_step: float | None = None,
        cg_max_step: float | None = None,
        cg_restart: int | None = None,
        # L-BFGS-B (scipy) on scalar cost with thickness bounds.
        lbfgs_m: int = 10,
        lbfgs_maxls: int = 20,
        # Bounded scipy trust-region reflective least squares.
        trf_x_scale: str | float = "jac",
        # Multi-start local search and optional automatic DE fallback.
        multistart_n: int = 8,
        multistart_method: str = "trf",
        multistart_seed: int | None = 0,
        auto_de_fallback: bool = True,
        auto_min_relative_improvement: float = 0.01,
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
        global_polish_method: str | None = None,
        da_initial_temp: float = 5230.0,
        da_visit: float = 2.62,
        da_accept: float = -5.0,
        # Local (LM/Adam) best checkpoints: every N iters, or every N epochs
        # for mini-batch Adam. Default: 5 (iters) / 1 (epochs if mini-batch).
        checkpoint_local_every: int | None = None,
        # Minimum free-layer thickness (metres). Unit in JSON configs: nm
        # via ``min_thickness_nm`` (default 8 nm). Raises per-material floors.
        min_thickness: float = 8e-9,
        # Optional per-material (lo, hi) bounds in metres. JSON configs use
        # ``thickness_bounds_nm`` and are converted before construction.
        thickness_bounds: dict[str, tuple[float, float]] | None = None,
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
        self.cg_initial_step = (
            float(adam_lr) if cg_initial_step is None else float(cg_initial_step)
        )
        self.cg_max_step = (
            float(adam_max_step) if cg_max_step is None else float(cg_max_step)
        )
        self.cg_restart = None if cg_restart is None else max(1, int(cg_restart))
        self.lbfgs_m = max(1, int(lbfgs_m))
        self.lbfgs_maxls = max(1, int(lbfgs_maxls))
        self.trf_x_scale = trf_x_scale
        self.multistart_n = max(1, int(multistart_n))
        self.multistart_method = str(multistart_method).lower().strip()
        self.multistart_seed = multistart_seed
        self.auto_de_fallback = bool(auto_de_fallback)
        self.auto_min_relative_improvement = max(
            0.0, float(auto_min_relative_improvement)
        )
        self.min_thickness = float(min_thickness)
        self.thickness_bounds = {
            str(mat).lower().replace("-", "_"): (float(bounds[0]), float(bounds[1]))
            for mat, bounds in (thickness_bounds or {}).items()
        }
        for mat, (lo, hi) in self.thickness_bounds.items():
            if lo < 0.0 or hi <= lo:
                raise ValueError(
                    f"thickness bound for {mat!r} requires 0 <= lower < upper"
                )
            if self.min_thickness >= hi:
                raise ValueError(
                    f"min_thickness must be below the upper bound for {mat!r}"
                )
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
        self.global_polish_method = resolve_global_polish_method(
            global_polish_method,
            global_polish_lm=global_polish_lm,
        )
        # Back-compat mirror of the resolved method.
        self.global_polish_lm = self.global_polish_method == "lm"
        self.da_initial_temp = float(da_initial_temp)
        self.da_visit = float(da_visit)
        self.da_accept = float(da_accept)
        if checkpoint_local_every is None:
            # Mini-batch: one checkpoint opportunity per epoch by default.
            # Full-grid Adam/LM: throttle to every 5 iterations.
            self.checkpoint_local_every = 1 if self.mini_batch else 5
        else:
            self.checkpoint_local_every = max(1, int(checkpoint_local_every))
        # Optional: on_best(layers, cost, info_dict) when the running best improves.
        self.on_best: Callable[..., None] | None = None
        self._last_notified_cost: float | None = None
        self._run_start_cost: float | None = None
        self._run_start_x: list[float] | None = None

    def _begin_run(self, x0: Sequence[float], start_cost: float) -> None:
        """Remember the run start for progress-only Δ logging."""
        self._run_start_cost = float(start_cost)
        self._run_start_x = [float(v) for v in x0]

    def _accept_best(
        self,
        cand_x: Sequence[float],
        cand_cost: float,
        best_x: Sequence[float],
        best_cost: float,
        x0: Sequence[float] | None = None,
    ) -> bool:
        # Thickness vectors remain in the signature for call-site stability;
        # checkpoint selection is deliberately based on loss alone.
        return is_better_checkpoint(cand_cost, best_cost)

    def _notify_best(
        self,
        layers: Sequence[tuple[str, float]],
        cost: float,
        *,
        stage: str,
        iter: int = 0,
        n_eval: int = 0,
        n_improve: int = 0,
        force: bool = False,
    ) -> None:
        """Invoke ``on_best`` when ``cost`` improves over the last checkpoint."""
        cb = self.on_best
        if cb is None:
            return
        if (
            not force
            and self._last_notified_cost is not None
            and cost >= self._last_notified_cost - 1e-15
        ):
            return
        self._last_notified_cost = float(cost)
        start_cost = self._run_start_cost
        delta = (
            float(cost) - float(start_cost) if start_cost is not None else 0.0
        )
        x = [float(d) for _, d in layers]
        x0 = self._run_start_x if self._run_start_x is not None else x
        cb(
            [(m, float(d)) for m, d in layers],
            float(cost),
            {
                "stage": stage,
                "iter": int(iter),
                "n_eval": int(n_eval),
                "n_improve": int(n_improve),
                "start_cost": start_cost,
                "delta": delta,
                "thickness_delta_nm": thickness_rms_nm(x, x0),
            },
        )

    def _make_result(
        self,
        *,
        best_layers: Sequence[tuple[str, float]],
        best_cost: float,
        best_r: Sequence[float],
        n_iter: int,
        success: bool,
        message: str,
        history: list[float],
        best_iter: int,
        final_layers: Sequence[tuple[str, float]],
        final_cost: float,
        start_cost: float,
        pre_polish: OptimResult | None = None,
    ) -> OptimResult:
        return OptimResult(
            [(m, float(d)) for m, d in best_layers],
            float(best_cost),
            list(best_r),
            int(n_iter),
            bool(success),
            message,
            list(history),
            best_iter=int(best_iter),
            pre_polish=pre_polish,
            final_layers=[(m, float(d)) for m, d in final_layers],
            final_cost=float(final_cost),
            start_cost=float(start_cost),
        )

    def _local_checkpoint_due(self, step: int, *, final: bool = False) -> bool:
        """True on epoch/iter boundaries for throttled local checkpoints."""
        if final:
            return True
        every = self.checkpoint_local_every
        return step > 0 and step % every == 0

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

    def bounds_for(self, material: str) -> tuple[float, float]:
        return _bounds_for(
            material,
            self.min_thickness,
            self.thickness_bounds,
        )

    def _project(self, materials: Sequence[str], x: list[float]) -> list[float]:
        out = []
        for mat, d in zip(materials, x):
            lo, hi = self.bounds_for(mat)
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
            lo, hi = self.bounds_for(materials[j])
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
            lo, hi = self.bounds_for(materials[j])
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
        # Allow the first point of this run to be checkpointed even if a prior
        # optimize() call already notified a similar cost.
        self._last_notified_cost = None
        if self.method in ("auto", "hybrid"):
            return self._optimize_auto(
                layers, free_indices=free_indices, verbose=verbose
            )
        if self.method in ("multistart", "multi_start", "multi-start"):
            return self._optimize_multistart(
                layers, free_indices=free_indices, verbose=verbose
            )
        if self.method in ("trf", "least_squares", "least-squares"):
            return self._optimize_trf(
                layers, free_indices=free_indices, verbose=verbose
            )
        if self.method == "adam":
            if self.mini_batch:
                return self._optimize_adam_minibatch(
                    layers, free_indices=free_indices, verbose=verbose
                )
            return self._optimize_adam(
                layers, free_indices=free_indices, verbose=verbose
            )
        if self.method in (
            "cg",
            "conjugate_gradient",
            "conjugate-gradient",
            "ncg",
        ):
            return self._optimize_cg(
                layers, free_indices=free_indices, verbose=verbose
            )
        if self.method in (
            "lbfgs",
            "l-bfgs",
            "lbfgsb",
            "l-bfgs-b",
        ):
            return self._optimize_lbfgs(
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
                "use 'auto', 'multistart', 'trf', 'lm', 'adam', 'cg', "
                "'lbfgs', 'de', or 'dual_annealing'"
            )
        return self._optimize_lm(layers, free_indices=free_indices, verbose=verbose)

    def _optimize_trf(
        self,
        layers: Sequence[tuple[str, float]],
        *,
        free_indices: Sequence[int] | None = None,
        verbose: bool = True,
    ) -> OptimResult:
        """Bounded trust-region reflective least squares.

        scipy receives thicknesses in nanometres rather than metres.  This
        keeps finite differences, trust-region radii and stopping tolerances
        on a well-conditioned numerical scale.
        """
        spo = _require_scipy()
        materials = [m for m, _ in layers]
        x0 = self._project(materials, [d for _, d in layers])
        free = list(range(len(x0))) if free_indices is None else list(free_indices)
        layers0 = list(zip(materials, x0))
        start_r = self.residuals(layers0)
        start_cost = 0.5 * sum(v * v for v in start_r)
        if not free:
            return self._make_result(
                best_layers=layers0,
                best_cost=start_cost,
                best_r=start_r,
                n_iter=0,
                success=False,
                message="no_free",
                history=[start_cost],
                best_iter=0,
                final_layers=layers0,
                final_cost=start_cost,
                start_cost=start_cost,
            )

        scale = 1e9
        y0 = [x0[j] * scale for j in free]
        lo = [self.bounds_for(materials[j])[0] * scale for j in free]
        hi = [self.bounds_for(materials[j])[1] * scale for j in free]

        def unpack(y: Sequence[float]) -> list[float]:
            x = list(x0)
            for value, j in zip(y, free):
                x[j] = float(value) / scale
            return self._project(materials, x)

        n_eval = 0

        def fun(y):
            nonlocal n_eval
            n_eval += 1
            return self.residuals(list(zip(materials, unpack(y))))

        def jac(y):
            nonlocal n_eval
            x = unpack(y)
            r0 = self.residuals(list(zip(materials, x)))
            full_j, free_cols = self._jacobian(materials, x, r0, free)
            n_eval += len(free_cols)
            # _jacobian differentiates with respect to metres; y is in nm.
            return [[row[j] / scale for j in free_cols] for row in full_j]

        if verbose:
            print(
                f"    TRF start: cost={start_cost:.6e}  layers={len(materials)}  "
                f"free={len(free)}  x_scale={self.trf_x_scale}",
                flush=True,
            )
        self._begin_run(x0, start_cost)
        self._notify_best(layers0, start_cost, stage="trf_start", iter=0)
        result = spo.least_squares(
            fun,
            y0,
            bounds=(lo, hi),
            method="trf",
            jac=jac,
            x_scale=self.trf_x_scale,
            ftol=max(self.tol, 1e-12),
            xtol=max(self.tol, 1e-12),
            gtol=max(self.tol, 1e-12),
            max_nfev=max(20, self.max_iter * (len(free) + 1)),
        )
        x_final = unpack(result.x)
        layers_final = list(zip(materials, x_final))
        final_r = self.residuals(layers_final)
        final_cost = 0.5 * sum(v * v for v in final_r)
        if final_cost < start_cost:
            best_layers, best_cost, best_r, best_iter = (
                layers_final,
                final_cost,
                final_r,
                1,
            )
            self._notify_best(
                best_layers,
                best_cost,
                stage="trf",
                iter=int(getattr(result, "nfev", n_eval)),
                n_eval=n_eval,
                n_improve=1,
            )
        else:
            best_layers, best_cost, best_r, best_iter = (
                layers0,
                start_cost,
                start_r,
                0,
            )
        if verbose:
            print(
                f"    TRF done: best={best_cost:.6e}  "
                f"Δ={best_cost-start_cost:+.3e}  nfev={n_eval}  "
                f"status={getattr(result, 'status', 0)}",
                flush=True,
            )
        return self._make_result(
            best_layers=best_layers,
            best_cost=best_cost,
            best_r=best_r,
            n_iter=int(getattr(result, "nfev", n_eval)),
            success=bool(result.success) or best_cost < start_cost,
            message=str(result.message),
            history=[start_cost, final_cost],
            best_iter=best_iter,
            final_layers=layers_final,
            final_cost=final_cost,
            start_cost=start_cost,
        )

    def _optimize_multistart(
        self,
        layers: Sequence[tuple[str, float]],
        *,
        free_indices: Sequence[int] | None = None,
        verbose: bool = True,
    ) -> OptimResult:
        """Latin-hypercube multi-start followed by a bounded local method."""
        materials = [m for m, _ in layers]
        x0 = self._project(materials, [d for _, d in layers])
        free = list(range(len(x0))) if free_indices is None else list(free_indices)
        if not free:
            return self._optimize_trf(
                layers, free_indices=free_indices, verbose=verbose
            )
        local = self.multistart_method
        if local not in ("trf", "lbfgs", "lm", "cg"):
            raise ValueError(
                f"multistart_method must be trf|lbfgs|lm|cg, got {local!r}"
            )

        rng = random.Random(self.multistart_seed)
        starts = [list(x0)]
        n_random = self.multistart_n - 1
        bins = list(range(n_random))
        per_dim_bins: dict[int, list[int]] = {}
        for j in free:
            shuffled = list(bins)
            rng.shuffle(shuffled)
            per_dim_bins[j] = shuffled
        for i in range(n_random):
            x = list(x0)
            for j in free:
                lo, hi = self.bounds_for(materials[j])
                u = (per_dim_bins[j][i] + rng.random()) / max(1, n_random)
                x[j] = lo + u * (hi - lo)
            starts.append(x)

        original_method = self.method
        original_callback = self.on_best
        self.on_best = None
        results: list[OptimResult] = []
        try:
            for i, start in enumerate(starts, 1):
                self.method = local
                if verbose:
                    print(
                        f"    multistart {i}/{len(starts)}: local={local}",
                        flush=True,
                    )
                results.append(
                    self.optimize(
                        list(zip(materials, start)),
                        free_indices=free,
                        verbose=False,
                    )
                )
        finally:
            self.method = original_method
            self.on_best = original_callback

        best = results[0]
        for candidate in results[1:]:
            if is_better_checkpoint(
                candidate.cost,
                best.cost,
            ):
                best = candidate
        start_cost = self.cost(list(zip(materials, x0)))
        history = [start_cost] + [r.cost for r in results]
        self._begin_run(x0, start_cost)
        self._notify_best(
            best.layers,
            best.cost,
            stage=f"multistart_{local}",
            iter=results.index(best) + 1,
            n_eval=sum(r.n_iter for r in results),
            n_improve=sum(r.cost < r.start_cost for r in results),
            force=True,
        )
        if verbose:
            print(
                f"    multistart done: starts={len(starts)}  "
                f"best={best.cost:.6e}  Δ={best.cost-start_cost:+.3e}",
                flush=True,
            )
        return self._make_result(
            best_layers=best.layers,
            best_cost=best.cost,
            best_r=best.residuals,
            n_iter=sum(r.n_iter for r in results),
            success=best.cost < start_cost,
            message=f"multistart({local}, n={len(starts)})",
            history=history,
            best_iter=history.index(min(history)),
            final_layers=best.final_layers or best.layers,
            final_cost=best.final_cost if best.final_cost is not None else best.cost,
            start_cost=start_cost,
        )

    def _optimize_auto(
        self,
        layers: Sequence[tuple[str, float]],
        *,
        free_indices: Sequence[int] | None = None,
        verbose: bool = True,
    ) -> OptimResult:
        """Multi-start local optimization with DE fallback when still poor."""
        start_cost = self.cost(layers)
        local = self._optimize_multistart(
            layers, free_indices=free_indices, verbose=verbose
        )
        _, _, specs_ok, _ = self.evaluate(local.layers)
        relative = (start_cost - local.cost) / max(abs(start_cost), 1e-15)
        if (
            not self.auto_de_fallback
            or (specs_ok and relative >= self.auto_min_relative_improvement)
        ):
            local.message = f"auto:{local.message}"
            return local

        if verbose:
            print(
                f"    auto fallback: specs_ok={specs_ok}  "
                f"relative_improvement={relative:.3%}; running DE",
                flush=True,
            )
        saved_method = self.method
        saved_polish = self.global_polish_method
        try:
            self.method = "de"
            self.global_polish_method = self.multistart_method
            global_result = self._optimize_differential_evolution(
                local.layers, free_indices=free_indices, verbose=verbose
            )
        finally:
            self.method = saved_method
            self.global_polish_method = saved_polish

        if is_better_checkpoint(global_result.cost, local.cost):
            winner = global_result
        else:
            winner = local
        winner.message = f"auto:{local.message}+de->{winner.message}"
        winner.start_cost = start_cost
        return winner

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
        bounds = [self.bounds_for(materials[j]) for j in free]
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
        self._begin_run(x0_full, start_cost)
        self._notify_best(
            start_layers,
            start_cost,
            stage=f"{tag}_start",
            iter=0,
            n_eval=0,
            n_improve=0,
        )

        def _sum_nm(x_vals: Sequence[float]) -> float:
            return sum(x_vals) * 1e9

        def objective(x_free) -> float:
            x = list(x0_full)
            for k, j in enumerate(free):
                x[j] = float(x_free[k])
            x = self._project(materials, x)
            layers_now = list(zip(materials, x))
            c = self.cost(layers_now)
            best["n_eval"] += 1
            if self._accept_best(x, c, best["x"], best["cost"]):
                best["cost"] = c
                best["x"] = list(x)
                best["n_improve"] += 1
                history.append(c)
                if verbose:
                    print(
                        f"    {tag} eval {best['n_eval']:5d}: "
                        f"NEW best={c:.6e}  "
                        f"Δ={c - start_cost:+.3e}  "
                        f"d_rms={thickness_rms_nm(x, x0_full):.2f} nm  "
                        f"Σd={_sum_nm(x):.1f} nm  "
                        f"improves={best['n_improve']}",
                        flush=True,
                    )
                self._notify_best(
                    layers_now,
                    c,
                    stage=tag,
                    iter=best["n_improve"],
                    n_eval=best["n_eval"],
                    n_improve=best["n_improve"],
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

    def _maybe_local_polish(
        self,
        layers: Sequence[tuple[str, float]],
        *,
        free_indices: Sequence[int] | None,
        history: list[float],
        verbose: bool,
        label: str,
        final_layers: Sequence[tuple[str, float]] | None = None,
        final_cost: float | None = None,
        start_cost: float | None = None,
    ) -> OptimResult:
        """Attach optional LM/Adam polish after a global (DE/DA) search."""
        cost = self.cost(layers)
        best_iter = (
            min(range(len(history)), key=lambda i: history[i]) if history else 0
        )
        sc = float(start_cost if start_cost is not None else history[0])
        fl = (
            [(m, float(d)) for m, d in final_layers]
            if final_layers is not None
            else [(m, float(d)) for m, d in layers]
        )
        fc = float(final_cost if final_cost is not None else cost)
        global_result = self._make_result(
            best_layers=layers,
            best_cost=cost,
            best_r=self.residuals(layers),
            n_iter=max(0, len(history) - 1),
            success=True,
            message=label,
            history=list(history),
            best_iter=best_iter,
            final_layers=fl,
            final_cost=fc,
            start_cost=sc,
        )
        polish = self.global_polish_method
        if polish == "none":
            return global_result

        # Preserve the global-run start for progress logging across polish.
        saved_start_x = list(self._run_start_x or [d for _, d in layers])
        saved_start_cost = self._run_start_cost

        if verbose:
            print(
                f"    {label}: local polish with {polish.upper()} …",
                flush=True,
            )
        if polish == "lm":
            polished = self._optimize_lm(
                layers, free_indices=free_indices, verbose=verbose
            )
        elif polish == "adam":
            # Full-grid Adam polish (ignore mini-batch settings).
            saved_mb = self.mini_batch
            self.mini_batch = False
            try:
                polished = self._optimize_adam(
                    layers, free_indices=free_indices, verbose=verbose
                )
            finally:
                self.mini_batch = saved_mb
        elif polish == "cg":
            polished = self._optimize_cg(
                layers, free_indices=free_indices, verbose=verbose
            )
        elif polish == "lbfgs":
            polished = self._optimize_lbfgs(
                layers, free_indices=free_indices, verbose=verbose
            )
        elif polish == "trf":
            polished = self._optimize_trf(
                layers, free_indices=free_indices, verbose=verbose
            )
        else:
            raise ValueError(f"internal: bad polish method {polish!r}")

        self._run_start_x = saved_start_x
        self._run_start_cost = saved_start_cost

        merged = list(history) + list(polished.history[1:])
        best_layers = list(polished.layers)
        best_cost = float(polished.cost)
        best_r = list(polished.residuals)
        merged_best_iter = (
            min(range(len(merged)), key=lambda i: merged[i]) if merged else 0
        )
        if self._accept_best(
            [d for _, d in global_result.layers],
            global_result.cost,
            [d for _, d in best_layers],
            best_cost,
            x0=saved_start_x,
        ):
            # Global stage has the lower loss.
            best_layers = list(global_result.layers)
            best_cost = float(global_result.cost)
            best_r = list(global_result.residuals)
        # Prefer polished final as the run's final iterate.
        pol_final_layers = polished.final_layers or polished.layers
        pol_final_cost = (
            polished.final_cost
            if polished.final_cost is not None
            else polished.cost
        )
        if verbose:
            print(
                f"    {label}+{polish}: "
                f"global_cost={global_result.cost:.6e} → "
                f"polished_cost={polished.cost:.6e}  "
                f"Δ={polished.cost - global_result.cost:+.3e}  "
                f"selected_cost={best_cost:.6e}",
                flush=True,
            )
        return self._make_result(
            best_layers=best_layers,
            best_cost=best_cost,
            best_r=best_r,
            n_iter=polished.n_iter + max(0, len(history) - 1),
            success=polished.success,
            message=f"{label}+{polish}",
            history=merged,
            best_iter=merged_best_iter,
            final_layers=pol_final_layers,
            final_cost=pol_final_cost,
            start_cost=sc,
            pre_polish=global_result,
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
        layers_final = list(zip(materials, x))
        result_cost = self.cost(layers_final)
        # Prefer the best point seen during search (polish can move off it).
        if best["cost"] <= result_cost + 1e-15:
            x_best = list(best["x"])
        else:
            x_best = list(x)
            best["cost"] = result_cost
        layers_best = list(zip(materials, x_best))
        final_cost = result_cost
        best_cost = self.cost(layers_best)
        if abs(best_cost - history[-1]) > 1e-15:
            history.append(best_cost)
        if verbose:
            free_nm = [round(x_best[j] * 1e9, 2) for j in free]
            print(
                f"    DE done: best={best_cost:.6e}  final={final_cost:.6e}  "
                f"Δ={best_cost - start_cost:+.3e}  "
                f"success={bool(result.success)}  "
                f"evals={best['n_eval']}  improves={best['n_improve']}  "
                f"Σd={sum_nm(x_best):.1f} nm",
                flush=True,
            )
            print(
                f"    DE done: free thicknesses (nm) = {free_nm}  "
                f"msg={result.message}",
                flush=True,
            )
        return self._maybe_local_polish(
            layers_best,
            free_indices=free_indices,
            history=history,
            verbose=verbose,
            label="de",
            final_layers=layers_final,
            final_cost=final_cost,
            start_cost=start_cost,
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
        layers_final = list(zip(materials, x))
        result_cost = self.cost(layers_final)
        if best["cost"] <= result_cost + 1e-15:
            x_best = list(best["x"])
        else:
            x_best = list(x)
        layers_best = list(zip(materials, x_best))
        final_cost = result_cost
        best_cost = self.cost(layers_best)
        if abs(best_cost - history[-1]) > 1e-15:
            history.append(best_cost)
        if verbose:
            free_nm = [round(x_best[j] * 1e9, 2) for j in free]
            print(
                f"    DA done: best={best_cost:.6e}  final={final_cost:.6e}  "
                f"Δ={best_cost - start_cost:+.3e}  "
                f"success={bool(result.success)}  "
                f"evals={best['n_eval']}  improves={best['n_improve']}  "
                f"Σd={sum_nm(x_best):.1f} nm  steps={step_count['n']}",
                flush=True,
            )
            print(
                f"    DA done: free thicknesses (nm) = {free_nm}  "
                f"msg={result.message}",
                flush=True,
            )
        return self._maybe_local_polish(
            layers_best,
            free_indices=free_indices,
            history=history,
            verbose=verbose,
            label="dual_annealing",
            final_layers=layers_final,
            final_cost=final_cost,
            start_cost=start_cost,
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
        layers_now = list(zip(materials, x))
        r = self.residuals(layers_now)
        cost = self.cost(layers_now)
        step_norm = math.sqrt(sum(d * d for d in delta))
        return x, cost, r, step_norm

    def _optimize_cg(
        self,
        layers: Sequence[tuple[str, float]],
        *,
        free_indices: Sequence[int] | None = None,
        verbose: bool = True,
    ) -> OptimResult:
        """Projected Polak–Ribière nonlinear CG on ``self.cost()``."""
        materials = [m for m, _ in layers]
        x = self._project(materials, [d for _, d in layers])
        free = list(range(len(x))) if free_indices is None else list(free_indices)
        layers0 = list(zip(materials, x))
        r = self.residuals(layers0)
        cost = self.cost(layers0)
        history = [cost]
        start_cost = cost
        x0 = list(x)
        best_x, best_cost, best_r, best_iter = list(x), cost, r, 0
        self._begin_run(x0, start_cost)

        restart_every = self.cg_restart if self.cg_restart is not None else max(1, len(free))
        alpha0 = max(self.cg_initial_step, 1e-15)
        max_step = max(self.cg_max_step, 1e-15)
        c1 = 1e-4

        g, free_cols = self._cost_gradient(materials, x, cost, free)
        if not free_cols:
            return self._make_result(
                best_layers=layers0,
                best_cost=cost,
                best_r=r,
                n_iter=0,
                success=False,
                message="no_free",
                history=history,
                best_iter=0,
                final_layers=layers0,
                final_cost=cost,
                start_cost=start_cost,
            )

        def _dot(a: Sequence[float], b: Sequence[float]) -> float:
            return sum(a[j] * b[j] for j in free_cols)

        def _neg_grad() -> list[float]:
            d = [0.0] * len(x)
            for j in free_cols:
                d[j] = -g[j]
            return d

        d = _neg_grad()
        gTg = _dot(g, g)

        if verbose:
            print(
                f"    CG start: cost={cost:.6e}  layers={len(materials)}  "
                f"free={len(free)}  step0={alpha0*1e9:.2f} nm  "
                f"restart={restart_every}",
                flush=True,
            )
        self._notify_best(
            list(zip(materials, x)),
            cost,
            stage="cg_start",
            iter=0,
            n_improve=0,
        )
        n_improve = 0
        pending_best = False

        for it in range(1, self.max_iter + 1):
            # Scale trial step so the largest free-component move ~ alpha0.
            d_abs = max((abs(d[j]) for j in free_cols), default=0.0)
            if d_abs <= 0.0 or gTg <= 0.0:
                break
            alpha = min(alpha0 / d_abs, max_step / d_abs)
            # Ensure descent after projection: require g·d < 0.
            gTd = _dot(g, d)
            if gTd >= 0.0:
                d = _neg_grad()
                gTd = _dot(g, d)
                d_abs = max((abs(d[j]) for j in free_cols), default=0.0)
                if d_abs <= 0.0 or gTd >= 0.0:
                    break
                alpha = min(alpha0 / d_abs, max_step / d_abs)

            accepted = False
            step_norm = 0.0
            for _ in range(20):
                x_trial = self._project(
                    materials, [x[i] + alpha * d[i] for i in range(len(x))]
                )
                layers_trial = list(zip(materials, x_trial))
                cost_trial = self.cost(layers_trial)
                # Armijo on projected step (use directional derivative of unprojected).
                if cost_trial <= cost + c1 * alpha * gTd:
                    delta = [x_trial[i] - x[i] for i in range(len(x))]
                    step_norm = math.sqrt(sum(v * v for v in delta))
                    x = x_trial
                    cost = cost_trial
                    r = self.residuals(layers_trial)
                    accepted = True
                    break
                alpha *= 0.5
                if alpha * d_abs < 1e-15:
                    break

            history.append(cost)
            if not accepted:
                # Restart as steepest descent with tiny step.
                d = _neg_grad()
                alpha = min(0.25 * alpha0 / max(d_abs, 1e-30), max_step / max(d_abs, 1e-30))
                x_trial = self._project(
                    materials, [x[i] + alpha * d[i] for i in range(len(x))]
                )
                layers_trial = list(zip(materials, x_trial))
                cost_trial = self.cost(layers_trial)
                if cost_trial < cost:
                    delta = [x_trial[i] - x[i] for i in range(len(x))]
                    step_norm = math.sqrt(sum(v * v for v in delta))
                    x, cost = x_trial, cost_trial
                    r = self.residuals(layers_trial)
                    history[-1] = cost
                else:
                    step_norm = 0.0

            if self._accept_best(x, cost, best_x, best_cost):
                best_x, best_cost, best_r, best_iter = list(x), cost, r, it
                n_improve += 1
                pending_best = True

            g_new, _ = self._cost_gradient(materials, x, cost, free)
            gTg_new = _dot(g_new, g_new)
            # Polak–Ribière (with automatic restart if β < 0).
            y_dot = sum((g_new[j] - g[j]) * g_new[j] for j in free_cols)
            beta = 0.0 if gTg <= 1e-30 else y_dot / gTg
            if beta < 0.0 or (it % restart_every) == 0:
                beta = 0.0
            d_new = [0.0] * len(x)
            for j in free_cols:
                d_new[j] = -g_new[j] + beta * d[j]
            g, gTg, d = g_new, gTg_new, d_new

            done = step_norm < 1e-12 or best_cost < self.tol or it == self.max_iter
            if pending_best and self._local_checkpoint_due(it, final=done):
                self._notify_best(
                    list(zip(materials, best_x)),
                    best_cost,
                    stage="cg",
                    iter=it,
                    n_improve=n_improve,
                )
                pending_best = False

            if verbose and (it == 1 or it % 5 == 0 or it == self.max_iter):
                print(
                    f"    CG iter {it:3d}: cost={cost:.6e}  "
                    f"best={best_cost:.6e}@iter{best_iter}  "
                    f"Δ={best_cost - start_cost:+.3e}  "
                    f"β={beta:.3f}  Σd={sum(best_x)*1e9:.1f} nm",
                    flush=True,
                )
            if step_norm < 1e-12 or best_cost < self.tol:
                if pending_best:
                    self._notify_best(
                        list(zip(materials, best_x)),
                        best_cost,
                        stage="cg",
                        iter=it,
                        n_improve=n_improve,
                    )
                return self._make_result(
                    best_layers=list(zip(materials, best_x)),
                    best_cost=best_cost,
                    best_r=best_r,
                    n_iter=it,
                    success=True,
                    message="converged",
                    history=history,
                    best_iter=best_iter,
                    final_layers=list(zip(materials, x)),
                    final_cost=cost,
                    start_cost=start_cost,
                )

        if pending_best:
            self._notify_best(
                list(zip(materials, best_x)),
                best_cost,
                stage="cg",
                iter=self.max_iter,
                n_improve=n_improve,
            )
        return self._make_result(
            best_layers=list(zip(materials, best_x)),
            best_cost=best_cost,
            best_r=best_r,
            n_iter=self.max_iter,
            success=best_cost < history[0],
            message="max_iter",
            history=history,
            best_iter=best_iter,
            final_layers=list(zip(materials, x)),
            final_cost=cost,
            start_cost=start_cost,
        )

    def _optimize_lbfgs(
        self,
        layers: Sequence[tuple[str, float]],
        *,
        free_indices: Sequence[int] | None = None,
        verbose: bool = True,
    ) -> OptimResult:
        """Bounded L-BFGS-B on ``self.cost()`` via scipy (needs scipy)."""
        spo = _require_scipy()
        materials = [m for m, _ in layers]
        x = self._project(materials, [d for _, d in layers])
        free = list(range(len(x))) if free_indices is None else list(free_indices)
        if not free:
            layers0 = list(zip(materials, x))
            c0 = self.cost(layers0)
            return self._make_result(
                best_layers=layers0,
                best_cost=c0,
                best_r=self.residuals(layers0),
                n_iter=0,
                success=False,
                message="no_free",
                history=[c0],
                best_iter=0,
                final_layers=layers0,
                final_cost=c0,
                start_cost=c0,
            )

        bounds = [self.bounds_for(materials[j]) for j in free]
        x0_free = [x[j] for j in free]
        layers0 = list(zip(materials, x))
        r = self.residuals(layers0)
        cost = self.cost(layers0)
        history = [cost]
        start_cost = cost
        x0 = list(x)
        best_x, best_cost, best_r, best_iter = list(x), cost, r, 0
        self._begin_run(x0, start_cost)
        n_improve = 0
        pending_best = False
        state = {"it": 0, "x": list(x), "cost": cost, "r": r}

        def _unpack(x_free: Sequence[float]) -> list[float]:
            xf = list(state["x"])
            for j, v in zip(free, x_free):
                xf[j] = float(v)
            return self._project(materials, xf)

        if verbose:
            print(
                f"    L-BFGS-B start: cost={cost:.6e}  layers={len(materials)}  "
                f"free={len(free)}  m={self.lbfgs_m}  maxiter={self.max_iter}",
                flush=True,
            )
        self._notify_best(
            list(zip(materials, x)),
            cost,
            stage="lbfgs_start",
            iter=0,
            n_improve=0,
        )

        def fun(x_free):
            x_full = _unpack(x_free)
            layers_now = list(zip(materials, x_full))
            c = self.cost(layers_now)
            return float(c)

        def jac(x_free):
            x_full = _unpack(x_free)
            layers_now = list(zip(materials, x_full))
            c = self.cost(layers_now)
            g, _ = self._cost_gradient(materials, x_full, c, free)
            return [g[j] for j in free]

        def callback(x_free, *_args):
            nonlocal best_x, best_cost, best_r, best_iter, n_improve, pending_best
            state["it"] += 1
            it = state["it"]
            x_full = _unpack(x_free)
            layers_now = list(zip(materials, x_full))
            c = self.cost(layers_now)
            rr = self.residuals(layers_now)
            state["x"] = list(x_full)
            state["cost"] = c
            state["r"] = rr
            history.append(c)
            if self._accept_best(x_full, c, best_x, best_cost):
                best_x, best_cost, best_r, best_iter = list(x_full), c, rr, it
                n_improve += 1
                pending_best = True
            done = it >= self.max_iter or best_cost < self.tol
            if pending_best and self._local_checkpoint_due(it, final=done):
                self._notify_best(
                    list(zip(materials, best_x)),
                    best_cost,
                    stage="lbfgs",
                    iter=it,
                    n_improve=n_improve,
                )
                pending_best = False
            if verbose and (it == 1 or it % 5 == 0 or it == self.max_iter):
                print(
                    f"    L-BFGS-B iter {it:3d}: cost={c:.6e}  "
                    f"best={best_cost:.6e}@iter{best_iter}  "
                    f"Δ={best_cost - start_cost:+.3e}  "
                    f"Σd={sum(best_x)*1e9:.1f} nm",
                    flush=True,
                )

        options = {
            "maxiter": int(self.max_iter),
            "maxfun": int(max(20, self.max_iter * 20)),
            "ftol": float(self.tol),
            "gtol": float(max(self.tol, 1e-12)),
            "maxcor": int(self.lbfgs_m),
            "maxls": int(self.lbfgs_maxls),
        }
        # Older scipy used factr; ftol is preferred on recent versions.
        try:
            result = spo.minimize(
                fun,
                x0_free,
                method="L-BFGS-B",
                jac=jac,
                bounds=bounds,
                callback=callback,
                options=options,
            )
        except TypeError:
            # Very old scipy callback signature / options differences.
            options.pop("ftol", None)
            options["factr"] = max(1.0, 1e2 / max(self.tol, 1e-15))
            result = spo.minimize(
                fun,
                x0_free,
                method="L-BFGS-B",
                jac=jac,
                bounds=bounds,
                callback=callback,
                options=options,
            )

        x_final = _unpack(result.x)
        layers_final = list(zip(materials, x_final))
        final_cost = self.cost(layers_final)
        final_r = self.residuals(layers_final)
        # Catch a last improvement not seen by callback.
        if self._accept_best(x_final, final_cost, best_x, best_cost):
            best_x, best_cost, best_r = list(x_final), final_cost, final_r
            best_iter = max(best_iter, state["it"])
            n_improve += 1
            pending_best = True

        n_iter = max(state["it"], int(getattr(result, "nit", 0) or 0))
        if pending_best:
            self._notify_best(
                list(zip(materials, best_x)),
                best_cost,
                stage="lbfgs",
                iter=n_iter,
                n_improve=n_improve,
            )

        msg = str(getattr(result, "message", "lbfgs"))
        if isinstance(msg, bytes):
            msg = msg.decode("utf-8", errors="replace")
        success = bool(getattr(result, "success", False)) or best_cost < start_cost
        return self._make_result(
            best_layers=list(zip(materials, best_x)),
            best_cost=best_cost,
            best_r=best_r,
            n_iter=n_iter,
            success=success,
            message=msg if msg else ("converged" if success else "lbfgs"),
            history=history,
            best_iter=best_iter,
            final_layers=layers_final,
            final_cost=final_cost,
            start_cost=start_cost,
        )

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
        layers0 = list(zip(materials, x))
        r = self.residuals(layers0)
        cost = self.cost(layers0)
        history = [cost]
        start_cost = cost
        x0 = list(x)
        best_x, best_cost, best_r, best_iter = list(x), cost, r, 0
        self._begin_run(x0, start_cost)

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
        self._notify_best(
            list(zip(materials, x)),
            cost,
            stage="adam_start",
            iter=0,
            n_improve=0,
        )
        n_improve = 0
        pending_best = False

        for it in range(1, self.max_iter + 1):
            x, cost, r, step_norm = self._adam_step(
                materials, x, free, m, v, lr=lr, t=it, cost=cost
            )
            history.append(cost)

            if self._accept_best(x, cost, best_x, best_cost):
                best_x, best_cost, best_r, best_iter = list(x), cost, r, it
                stale = 0
                n_improve += 1
                pending_best = True
            else:
                stale += 1
                if stale >= 8:
                    lr = max(lr * 0.5, 0.05e-9)
                    stale = 0

            done = step_norm < 1e-12 or best_cost < self.tol or it == self.max_iter
            if pending_best and self._local_checkpoint_due(it, final=done):
                self._notify_best(
                    list(zip(materials, best_x)),
                    best_cost,
                    stage="adam",
                    iter=it,
                    n_improve=n_improve,
                )
                pending_best = False

            if verbose and (it == 1 or it % 5 == 0 or it == self.max_iter):
                print(
                    f"    Adam iter {it:3d}: cost={cost:.6e}  "
                    f"best={best_cost:.6e}@iter{best_iter}  "
                    f"Δ={best_cost - start_cost:+.3e}  "
                    f"lr={lr*1e9:.2f} nm  Σd={sum(best_x)*1e9:.1f} nm",
                    flush=True,
                )
            if step_norm < 1e-12 or best_cost < self.tol:
                if pending_best:
                    self._notify_best(
                        list(zip(materials, best_x)),
                        best_cost,
                        stage="adam",
                        iter=it,
                        n_improve=n_improve,
                    )
                return self._make_result(
                    best_layers=list(zip(materials, best_x)),
                    best_cost=best_cost,
                    best_r=best_r,
                    n_iter=it,
                    success=True,
                    message="converged",
                    history=history,
                    best_iter=best_iter,
                    final_layers=list(zip(materials, x)),
                    final_cost=cost,
                    start_cost=start_cost,
                )

        if pending_best:
            self._notify_best(
                list(zip(materials, best_x)),
                best_cost,
                stage="adam",
                iter=self.max_iter,
                n_improve=n_improve,
            )
        return self._make_result(
            best_layers=list(zip(materials, best_x)),
            best_cost=best_cost,
            best_r=best_r,
            n_iter=self.max_iter,
            success=best_cost < history[0],
            message="max_iter",
            history=history,
            best_iter=best_iter,
            final_layers=list(zip(materials, x)),
            final_cost=cost,
            start_cost=start_cost,
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
        layers0 = list(zip(materials, x))
        r = self.residuals(layers0)
        full_cost = self.cost(layers0)
        history = [full_cost]
        start_cost = full_cost
        x0 = list(x)
        best_x, best_cost, best_r, best_iter = list(x), full_cost, r, 0
        self._begin_run(x0, start_cost)

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
        self._notify_best(
            list(zip(materials, x)),
            full_cost,
            stage="adam_minibatch_start",
            iter=0,
            n_improve=0,
        )
        pending_best = False
        n_improve = 0

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
                batch_cost = self.cost(batch_layers)
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
            layers_now = list(zip(materials, x))
            r = self.residuals(layers_now)
            full_cost = self.cost(layers_now)
            history.append(full_cost)

            if self._accept_best(x, full_cost, best_x, best_cost):
                best_x, best_cost, best_r, best_iter = list(x), full_cost, r, epoch
                stale = 0
                n_improve += 1
                pending_best = True
            else:
                stale += 1
                if stale >= 8:
                    lr = max(lr * 0.5, 0.05e-9)
                    stale = 0

            done = best_cost < self.tol or epoch == n_epochs
            if pending_best and self._local_checkpoint_due(epoch, final=done):
                self._notify_best(
                    list(zip(materials, best_x)),
                    best_cost,
                    stage="adam_minibatch",
                    iter=epoch,
                    n_improve=n_improve,
                )
                pending_best = False

            if verbose:
                mean_batch = epoch_batch_cost_sum / max(1, n_batches)
                print(
                    f"    epoch {epoch:3d} done: full_cost={full_cost:.6e}  "
                    f"mean_batch_cost={mean_batch:.6e}  "
                    f"best={best_cost:.6e}@epoch{best_iter}  "
                    f"Δ={best_cost - start_cost:+.3e}  "
                    f"lr={lr*nm:.2f} nm  Σd={sum(best_x)*nm:.1f} nm",
                    flush=True,
                )
            if best_cost < self.tol:
                if pending_best:
                    self._notify_best(
                        list(zip(materials, best_x)),
                        best_cost,
                        stage="adam_minibatch",
                        iter=epoch,
                        n_improve=n_improve,
                    )
                self._set_wavelengths(full_wls)
                return self._make_result(
                    best_layers=list(zip(materials, best_x)),
                    best_cost=best_cost,
                    best_r=best_r,
                    n_iter=epoch,
                    success=True,
                    message="converged",
                    history=history,
                    best_iter=best_iter,
                    final_layers=list(zip(materials, x)),
                    final_cost=full_cost,
                    start_cost=start_cost,
                )

        if pending_best:
            self._notify_best(
                list(zip(materials, best_x)),
                best_cost,
                stage="adam_minibatch",
                iter=n_epochs,
                n_improve=n_improve,
            )
        self._set_wavelengths(full_wls)
        return self._make_result(
            best_layers=list(zip(materials, best_x)),
            best_cost=best_cost,
            best_r=best_r,
            n_iter=n_epochs,
            success=best_cost < history[0],
            message="max_epochs",
            history=history,
            best_iter=best_iter,
            final_layers=list(zip(materials, x)),
            final_cost=full_cost,
            start_cost=start_cost,
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
        start_cost = cost
        x0 = list(x)
        best_x, best_cost, best_r, best_iter = list(x), cost, r, 0
        self._begin_run(x0, start_cost)

        if verbose:
            print(
                f"    LM start: cost={cost:.6e}  layers={len(materials)}  "
                f"free={len(free)}",
                flush=True,
            )
        self._notify_best(
            list(zip(materials, x)),
            cost,
            stage="lm_start",
            iter=0,
            n_improve=0,
        )
        n_improve = 0
        pending_best = False

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
                if self._accept_best(x, cost, best_x, best_cost):
                    best_x, best_cost, best_r, best_iter = list(x), cost, r, it
                    n_improve += 1
                    pending_best = True
                lam = max(lam * 0.3, 1e-8)
                done = (
                    step_norm < 1e-12
                    or best_cost < self.tol
                    or it == self.max_iter
                )
                if pending_best and self._local_checkpoint_due(it, final=done):
                    self._notify_best(
                        list(zip(materials, best_x)),
                        best_cost,
                        stage="lm",
                        iter=it,
                        n_improve=n_improve,
                    )
                    pending_best = False
                if verbose and (it == 1 or it % 5 == 0 or it == self.max_iter):
                    total_nm = sum(best_x) * 1e9
                    print(
                        f"    LM iter {it:3d}: cost={cost:.6e}  "
                        f"best={best_cost:.6e}@iter{best_iter}  "
                        f"Δ={best_cost - start_cost:+.3e}  "
                        f"λ={lam:.2e}  Σd={total_nm:.1f} nm",
                        flush=True,
                    )
                if step_norm < 1e-12 or best_cost < self.tol:
                    if pending_best:
                        self._notify_best(
                            list(zip(materials, best_x)),
                            best_cost,
                            stage="lm",
                            iter=it,
                            n_improve=n_improve,
                        )
                    return self._make_result(
                        best_layers=list(zip(materials, best_x)),
                        best_cost=best_cost,
                        best_r=best_r,
                        n_iter=it,
                        success=True,
                        message="converged",
                        history=history,
                        best_iter=best_iter,
                        final_layers=list(zip(materials, x)),
                        final_cost=cost,
                        start_cost=start_cost,
                    )
            else:
                lam = min(lam * 3.0, 1e8)

        if pending_best:
            self._notify_best(
                list(zip(materials, best_x)),
                best_cost,
                stage="lm",
                iter=self.max_iter,
                n_improve=n_improve,
            )
        return self._make_result(
            best_layers=list(zip(materials, best_x)),
            best_cost=best_cost,
            best_r=best_r,
            n_iter=self.max_iter,
            success=best_cost < history[0],
            message="max_iter",
            history=history,
            best_iter=best_iter,
            final_layers=list(zip(materials, x)),
            final_cost=cost,
            start_cost=start_cost,
        )

    def evaluate(self, layers: Sequence[tuple[str, float]]):
        R, T = self._rt(layers)
        ok = specs_satisfied(layers, self.bands, self.wavelengths, R, T)
        return R, T, ok, band_report(self.bands, self.wavelengths, R, T)
