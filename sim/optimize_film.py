"""Bounded thickness optimization and optional Needle synthesis for text stacks.

Initial stack format matches ``plot_rt_txt.py``. Optimises **coating** layer
thicknesses only; the first row (incident medium) and last row (substrate)
are never free variables — their thicknesses stay exactly as in the input.

Optical constants: by default ``nk_source=library`` loads n(λ),k(λ) from the
dispersion database by material name. Set ``nk_source=fixed`` (or
``use_fixed_nk: true``) to use the constant n,k columns from the stack file.

Initial thicknesses: stack-file values by default. Set ``init.mode`` to
``random_qw`` (quarter-wave + jitter over the band span) or
``random_uniform``, or ``random_init: true``, to draw a reproducible random
guess while keeping the material sequence.

Objective: build a piecewise R target per band, then minimize one shared
band-normalized least-squares merit plus an optional thickness penalty::

    L = Σ_b w_b · mean_{λ∈b} (R − t_b)² / Σ_b w_b
        + thickness_weight · (Σ d / d_ref)²

Each band sets ``t_b`` via ``R_target`` (0–1). If omitted, ``objective:
maximize|minimize`` defaults to ``t_b = 1`` or ``0``. Independent of sample
count and absolute weight scale.

Usage::

    python3 sim/optimize_film.py \\
        sim/examples/example_stack.txt \\
        sim/examples/example_optimize_film.json

The default ``method=auto`` runs Latin-hypercube multi-start TRF, then uses
DE with local polish if the local stage remains infeasible or barely improves.
Set ``use_needle=true`` to allow sensitivity-guided layer insertion and pruning.

Adam training modes:

  - **full grid** (default): each iteration uses the full wavelength grid.
  - **mini-batch**: set ``mini_batch`` true or to an object with
    ``batch_size``, ``n_batches``, optional ``n_epochs`` / ``shuffle_seed``.
    Each epoch draws ``n_batches`` batches; every batch picks a random float
    start in ``[λ_min, λ_max)`` and takes ``batch_size`` points spaced by
    ``(λ_max−λ_min)/batch_size``, wrapping at ``λ_max`` back to ``λ_min``.
    One Adam step per batch; full-grid cost is recorded per epoch.

Other local methods: ``method=trf`` (bounded trust-region least squares),
``method=lm`` (projected Gauss–Newton on residuals),
``method=cg`` (Polak–Ribière nonlinear CG on the scalar loss),
``method=lbfgs`` (scipy L-BFGS-B on the scalar loss; needs scipy).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tmm
import dispersion as dsp
from lm_optimizer import (
    BandSpec,
    LMThicknessOptimizer,
    _bounds_for,
    parse_multistart_sampling_bounds_nm,
    parse_thickness_bounds_nm,
)
from needle import NeedleSynthesizer
from plot_rt import (
    close_all_figures,
    configure_matplotlib,
    dense_grid_nm,
    materials_used_in_stack,
    plot_results,
    plot_rt,
    plot_used_nk,
    sample_nk_curves,
    write_band_stats_csv,
    write_nk_csv,
    write_spectrum_csv,
)
from plot_rt_txt import StackRow, load_stack_txt
from rt_calculator import DEFAULT_SUBSTRATE_THICKNESS, RTCalculator, make_calculator

_NM = 1e-9
_NK_REF_NM_DEFAULT = 550.0


def _design_wavelengths_from_bands(
    bands: Sequence[RObjectiveBand] | Sequence[BandSpec],
    *,
    n_centres: int = 6,
) -> list[float]:
    """Log-spaced centres spanning all band wavelengths (metres)."""
    if not bands:
        return [550e-9, 1000e-9, 1500e-9]
    lo = min(float(b.wl_lo) for b in bands)
    hi = max(float(b.wl_hi) for b in bands)
    if hi <= lo:
        return [lo]
    n = max(1, int(n_centres))
    if n == 1:
        return [math.sqrt(lo * hi)]
    log_lo, log_hi = math.log(lo), math.log(hi)
    return [
        math.exp(log_lo + (i + 0.5) * (log_hi - log_lo) / n) for i in range(n)
    ]


def random_qw_init(
    layers: Sequence[tuple[str, float]],
    bands: Sequence[RObjectiveBand] | Sequence[BandSpec],
    *,
    seed: int | None = None,
    jitter: float = 0.35,
    qw_fraction: float = 0.25,
    min_thickness_m: float = 8e-9,
    thickness_bounds: dict[str, tuple[float, float]] | None = None,
) -> list[tuple[str, float]]:
    """Physically reasonable random thicknesses; keep material sequence.

    Each layer is initialised near a quarter-wave optical thickness at a
    randomly chosen design wavelength from the band span, with relative
    multiplicative jitter, then projected onto material bounds.
    """
    rng = random.Random(seed)
    centres = _design_wavelengths_from_bands(bands)
    j = max(0.0, float(jitter))
    frac = max(0.05, float(qw_fraction))
    out: list[tuple[str, float]] = []
    for mat, _d0 in layers:
        key = dsp.normalize_material_name(mat)
        wl = centres[rng.randrange(len(centres))]
        try:
            n = float(dsp.material_n(key, wl).real)
        except Exception:
            n = 1.5
        n = max(n, 1.01)
        scale = math.exp(rng.uniform(-j, j)) if j > 0 else 1.0
        d = frac * wl / n * scale
        lo, hi = _bounds_for(key, min_thickness_m, thickness_bounds)
        out.append((key, min(hi, max(lo, d))))
    return out


def apply_init_layers(
    layers: Sequence[tuple[str, float]],
    bands: Sequence[RObjectiveBand],
    cfg: dict,
) -> tuple[list[tuple[str, float]], str]:
    """Apply JSON ``init`` / ``random_init`` settings; return (layers, note)."""
    init_cfg = cfg.get("init")
    if not isinstance(init_cfg, dict):
        init_cfg = {}
    mode = str(
        init_cfg.get(
            "mode",
            "random_qw" if bool(cfg.get("random_init", False)) else "keep",
        )
    ).lower().strip()
    if mode in ("keep", "file", "none", "off", "false", "0"):
        return [tuple(x) for x in layers], "keep (stack file thicknesses)"

    seed = init_cfg.get("seed", cfg.get("init_seed", cfg.get("global_seed")))
    if seed is not None:
        seed = int(seed)
    jitter = float(init_cfg.get("jitter", cfg.get("init_jitter", 0.35)))
    qw_fraction = float(init_cfg.get("qw_fraction", 0.25))
    min_nm = float(cfg.get("min_thickness_nm", 8.0))
    thickness_bounds = parse_thickness_bounds_nm(cfg.get("thickness_bounds_nm"))

    if mode in ("random_qw", "qw", "random", "true", "1"):
        layers1 = random_qw_init(
            layers,
            bands,
            seed=seed,
            jitter=jitter,
            qw_fraction=qw_fraction,
            min_thickness_m=min_nm * _NM,
            thickness_bounds=thickness_bounds,
        )
        note = (
            f"random_qw  seed={seed}  jitter={jitter:g}  "
            f"qw_fraction={qw_fraction:g}"
        )
        return layers1, note

    if mode in ("random_uniform", "uniform"):
        rng = random.Random(seed)
        out: list[tuple[str, float]] = []
        for mat, _d0 in layers:
            key = dsp.normalize_material_name(mat)
            lo, hi = _bounds_for(key, min_nm * _NM, thickness_bounds)
            # Prefer mid-thin range: log-uniform between lo and min(hi, 200 nm).
            hi_eff = min(hi, 200e-9)
            if hi_eff <= lo:
                d = lo
            else:
                d = math.exp(rng.uniform(math.log(lo), math.log(hi_eff)))
            out.append((key, d))
        note = f"random_uniform  seed={seed}"
        return out, note

    raise ValueError(
        f"unknown init.mode {mode!r}; use keep|random_qw|random_uniform"
    )


def parse_nk_source(cfg: dict) -> str:
    """Return ``'library'`` (dispersion DB) or ``'fixed'`` (stack-file n,k).

    Default is ``library``. Explicit knobs:

    - ``nk_source``: ``library`` | ``fixed`` (aliases: lib/dispersion/database,
      stack/constant/input)
    - ``use_fixed_nk``: true → fixed, false → library
    - ``use_library_nk``: true → library, false → fixed
    """
    raw = cfg.get("nk_source")
    if raw is not None:
        s = str(raw).strip().lower()
        aliases = {
            "library": "library",
            "lib": "library",
            "dispersion": "library",
            "database": "library",
            "db": "library",
            "fixed": "fixed",
            "stack": "fixed",
            "constant": "fixed",
            "input": "fixed",
            "file": "fixed",
        }
        if s not in aliases:
            raise ValueError(
                f"nk_source must be library|fixed (got {raw!r}); "
                f"known: {sorted(set(aliases))}"
            )
        return aliases[s]
    if "use_fixed_nk" in cfg:
        return "fixed" if bool(cfg["use_fixed_nk"]) else "library"
    if "use_library_nk" in cfg:
        return "library" if bool(cfg["use_library_nk"]) else "fixed"
    return "library"


def library_nk_table(
    materials: Sequence[str],
    *,
    ref_wavelength_m: float,
) -> dict[str, complex]:
    """Look up N at ``ref_wavelength_m`` for stack-file export / display."""
    table: dict[str, complex] = {}
    for name in materials:
        key = dsp.normalize_material_name(name)
        if key not in dsp.MATERIALS:
            raise KeyError(
                f"material {name!r} not in dispersion library; "
                f"known: {sorted(dsp.MATERIALS)}"
            )
        table[key] = dsp.material_n(key, ref_wavelength_m)
        # Also index by original lower-case token if different (hyphen forms).
        low = str(name).strip().lower()
        if low not in table:
            table[low] = table[key]
    return table


def require_library_materials(*names: str) -> None:
    missing = []
    for name in names:
        key = dsp.normalize_material_name(name)
        if key not in dsp.MATERIALS:
            missing.append(name)
    if missing:
        raise KeyError(
            f"nk_source=library but material(s) missing from database: "
            f"{missing}; known: {sorted(dsp.MATERIALS)}"
        )


@dataclass
class RObjectiveBand:
    """One wavelength interval with a reflectance target ``r_target``."""

    wl_lo: float
    wl_hi: float
    maximize: bool
    weight: float = 1.0
    r_target: float = 0.0

    def as_band_spec(self) -> BandSpec:
        """Map to BandSpec for grid / shaded plot targets."""
        t = float(self.r_target)
        if self.maximize:
            return BandSpec(
                wl_lo=self.wl_lo,
                wl_hi=self.wl_hi,
                R_min=min(0.9, max(0.0, t - 0.05)),
                weight=self.weight,
                R_target=t,
            )
        return BandSpec(
            wl_lo=self.wl_lo,
            wl_hi=self.wl_hi,
            R_max=max(0.1, min(1.0, t + 0.05)),
            weight=self.weight,
            R_target=t,
        )


def _parse_objective(raw: str) -> bool:
    s = str(raw).strip().lower()
    if s in ("max", "maximize", "maximum", "high", "reflect"):
        return True
    if s in ("min", "minimize", "minimum", "low", "antireflect"):
        return False
    raise ValueError(
        f"band objective must be maximize/minimize (or max/min), got {raw!r}"
    )


def parse_r_bands(cfg: dict) -> list[RObjectiveBand]:
    raw = cfg.get("bands")
    if not isinstance(raw, list) or not raw:
        raise ValueError("config needs a non-empty 'bands' list")
    n_declare = cfg.get("n_bands")
    if n_declare is not None and int(n_declare) != len(raw):
        raise ValueError(
            f"n_bands={n_declare} does not match len(bands)={len(raw)}"
        )
    out: list[RObjectiveBand] = []
    for i, b in enumerate(raw):
        w = b.get("wavelength_nm")
        if not w or len(w) != 2:
            raise ValueError(f"bands[{i}]: need wavelength_nm: [lo, hi]")
        lo, hi = float(w[0]), float(w[1])
        if hi <= lo:
            raise ValueError(f"bands[{i}]: invalid range {lo}–{hi} nm")

        obj = b.get("objective", b.get("R", b.get("goal")))
        t_raw = b.get("R_target", b.get("r_target", b.get("target")))
        if obj is None and t_raw is None:
            raise ValueError(
                f"bands[{i}]: need objective maximize|minimize and/or "
                "R_target in [0, 1]"
            )

        if t_raw is not None:
            t = float(t_raw)
            if not (0.0 <= t <= 1.0):
                raise ValueError(
                    f"bands[{i}]: R_target must be in [0, 1], got {t_raw!r}"
                )
        else:
            t = 1.0 if _parse_objective(obj) else 0.0

        if obj is not None:
            maximize = _parse_objective(obj)
        else:
            maximize = t >= 0.5

        out.append(
            RObjectiveBand(
                wl_lo=lo * _NM,
                wl_hi=hi * _NM,
                maximize=maximize,
                weight=float(b.get("weight", 1.0)),
                r_target=t,
            )
        )
    return out


class ConstantNkCalculator(RTCalculator):
    """TMM using fixed N=n+ik from the text stack (no dispersion library)."""

    def __init__(self, nk: dict[str, complex], *, use_cuda: bool = False):
        self.nk = {name.lower(): complex(val) for name, val in nk.items()}
        self.use_cuda = bool(use_cuda)
        if self.use_cuda:
            import tmm_cuda

            tmm_cuda.require_cupy()

    def _N(self, name: str) -> complex:
        key = name.lower()
        if key not in self.nk:
            raise KeyError(
                f"unknown material {name!r}; stack provides: {sorted(self.nk)}"
            )
        return self.nk[key]

    def spectrum(
        self,
        layers: Sequence[tuple[str, float]],
        wavelengths: Sequence[float],
        theta0: float = 0.0,
        *,
        incident: str = "air",
        substrate: str = "glass",
        substrate_thickness: float = DEFAULT_SUBSTRATE_THICKNESS,
        exit_medium: str = "air",
        polarization: str = "unpolarized",
        substrate_model: str = "semi_infinite",
    ) -> tuple[list[float], list[float]]:
        del substrate_thickness, exit_medium, substrate_model  # semi-infinite only
        n_inc = self._N(incident)
        n_sub = self._N(substrate)
        coating = [(self._N(m), d) for m, d in layers]
        if self.use_cuda:
            import tmm_cuda

            n_list = [n_inc, *[n for n, _ in coating], n_sub]
            d_list = [0.0, *[d for _, d in coating], 0.0]
            return tmm_cuda.spectrum_constant_n(
                n_list, d_list, wavelengths, theta0, polarization
            )
        rs, ts = [], []
        for wl in wavelengths:
            stack = [(n_inc, 0.0), *coating, (n_sub, 0.0)]
            if polarization in ("unpolarized", "avg", "average"):
                r, t = tmm.unpolarised(tmm.coherent_rt, stack, wl, theta0)
            elif polarization in ("s", "p"):
                r, t = tmm.coherent_rt(stack, wl, theta0, polarization)
            else:
                raise ValueError(
                    f"polarization must be s/p/unpolarized, got {polarization!r}"
                )
            rs.append(r)
            ts.append(t)
        return rs, ts


def build_r_target_curve(
    bands: Sequence[RObjectiveBand],
    wavelengths: Sequence[float],
) -> tuple[list[float | None], list[float]]:
    """Piecewise R target from each band's ``r_target``; None outside bands.

    If bands overlap, the last listed band wins for that wavelength.
    """
    targets: list[float | None] = [None] * len(wavelengths)
    weights = [0.0] * len(wavelengths)
    for b in bands:
        t = float(b.r_target)
        w = max(float(b.weight), 0.0)
        for i, wl in enumerate(wavelengths):
            if b.wl_lo - 1e-15 <= wl <= b.wl_hi + 1e-15:
                targets[i] = t
                weights[i] = w
    return targets, weights


def _band_sample_indices(
    bands: Sequence[RObjectiveBand],
    wavelengths: Sequence[float],
) -> list[list[int]]:
    """Sorted wavelength indices belonging to each band."""
    out: list[list[int]] = []
    for b in bands:
        idx = [
            i
            for i, wl in enumerate(wavelengths)
            if b.wl_lo - 1e-15 <= wl <= b.wl_hi + 1e-15
        ]
        out.append(idx)
    return out


def _active_band_weight_sum(
    bands: Sequence[RObjectiveBand],
    band_indices: Sequence[Sequence[int]],
) -> float:
    """Sum of weights over bands that have at least one sample."""
    return sum(
        max(float(b.weight), 0.0)
        for b, idx in zip(bands, band_indices)
        if idx and b.weight > 0
    )


def reflectance_rmse(
    R: Sequence[float],
    bands: Sequence[RObjectiveBand],
    wavelengths: Sequence[float],
) -> float:
    """Band-normalized RMSE, independent of N and absolute weight scale.

    For each band ``b`` compute the in-band mean squared error, then

        MSE  = Σ_b w_b · mean_b (R − t_b)² / Σ_b w_b
        RMSE = sqrt(MSE)

    So widening a band (more sample points) or scaling all ``w_b`` by a
    constant leaves the value unchanged; only *relative* band weights matter.
    """
    band_indices = _band_sample_indices(bands, wavelengths)
    w_sum = _active_band_weight_sum(bands, band_indices)
    if w_sum <= 0.0:
        return 0.0
    mse = 0.0
    for b, idx in zip(bands, band_indices):
        w = max(float(b.weight), 0.0)
        if not idx or w <= 0.0:
            continue
        t = float(b.r_target)
        mean_sq = sum((R[i] - t) ** 2 for i in idx) / len(idx)
        mse += (w / w_sum) * mean_sq
    return math.sqrt(mse)


def thickness_penalty(
    layers: Sequence[tuple[str, float]],
    *,
    thickness_weight: float,
    thickness_ref: float = 1000e-9,
) -> float:
    """``thickness_weight * (Σ d / d_ref)²``; 0 when weight ≤ 0 or no layers."""
    if thickness_weight <= 0 or not layers:
        return 0.0
    total = sum(d for _, d in layers)
    return float(thickness_weight) * (total / thickness_ref) ** 2


def build_maxmin_r_residuals(
    layers: Sequence[tuple[str, float]],
    bands: Sequence[RObjectiveBand],
    wavelengths: Sequence[float],
    R: Sequence[float],
    *,
    thickness_weight: float = 0.0,
    thickness_ref: float = 1000e-9,
) -> list[float]:
    """Residuals for band-normalized L2 fit + thickness.

    Fit residuals are scaled so ``0.5 * sum(r**2)`` equals the band-normalized
    MSE (= RMSE²). Thickness residual matches ``thickness_penalty``.
    """
    band_indices = _band_sample_indices(bands, wavelengths)
    w_sum = _active_band_weight_sum(bands, band_indices)
    if w_sum <= 0.0:
        res = [0.0]
    else:
        res = []
        for b, idx in zip(bands, band_indices):
            w = max(float(b.weight), 0.0)
            if not idx or w <= 0.0:
                continue
            t = float(b.r_target)
            n = len(idx)
            # 0.5 Σ_i r_i² = (w/W) · mean (R−t)²
            scale = math.sqrt(2.0 * w / (w_sum * n))
            for i in idx:
                res.append(scale * (R[i] - t))

    if thickness_weight > 0 and layers:
        total = sum(d for _, d in layers)
        res.append(math.sqrt(2.0 * thickness_weight) * total / thickness_ref)
    if not res:
        res = [0.0]
    return res


class MaxMinROptimizer(LMThicknessOptimizer):
    """Thickness optimiser minimizing one shared residual least-squares merit."""

    def __init__(
        self,
        calculator: RTCalculator,
        rbands: Sequence[RObjectiveBand],
        **kwargs,
    ):
        self.rbands = list(rbands)
        super().__init__(
            calculator,
            [b.as_band_spec() for b in rbands],
            **kwargs,
        )
        self._targets, self._target_weights = build_r_target_curve(
            self.rbands, self.wavelengths
        )

    def residuals(self, layers: Sequence[tuple[str, float]]) -> list[float]:
        R, _T = self._rt(layers)
        return build_maxmin_r_residuals(
            layers,
            self.rbands,
            self.wavelengths,
            R,
            thickness_weight=self.thickness_weight,
        )

    def cost(self, layers: Sequence[tuple[str, float]]) -> float:
        """Use exactly the merit optimized by LM/TRF and scalar methods."""
        r = self.residuals(layers)
        return 0.5 * sum(v * v for v in r)


def nk_table(
    incident: StackRow, films: list[StackRow], substrate: StackRow
) -> dict[str, complex]:
    table: dict[str, complex] = {}
    for row in (incident, *films, substrate):
        key = row.material.lower()
        N = row.N
        if key in table and abs(table[key] - N) > 1e-9:
            raise ValueError(
                f"material {row.material!r} appears with inconsistent n,k "
                f"({table[key]} vs {N})"
            )
        table[key] = N
    return table


def write_stack_txt(
    path: str,
    incident: StackRow,
    films: list[tuple[str, float]],
    substrate: StackRow,
    nk: dict[str, complex],
    *,
    film_indices: Sequence[int] | None = None,
    header_lines: list[str] | None = None,
) -> None:
    """Write stack text; incident/substrate thicknesses kept from input rows."""

    def _N(name: str) -> complex:
        key = dsp.normalize_material_name(name)
        if key in nk:
            return nk[key]
        low = str(name).strip().lower()
        if low in nk:
            return nk[low]
        raise KeyError(f"no n,k for material {name!r} in nk table")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines: list[str] = []
    if header_lines:
        lines.extend(header_lines)
    lines.append("# index  material  thickness_nm  n  k")
    Ni = _N(incident.material)
    lines.append(
        f"{incident.index}  {incident.material}  {incident.thickness_nm:.6g}  "
        f"{Ni.real:.6g}  {Ni.imag:.6g}"
    )
    for i, (mat, d) in enumerate(films):
        idx = film_indices[i] if film_indices is not None else i + 1
        N = _N(mat)
        lines.append(
            f"{idx}  {mat}  {d / _NM:.4f}  {N.real:.6g}  {N.imag:.6g}"
        )
    Ns = _N(substrate.material)
    substrate_index = substrate.index if film_indices is not None else len(films) + 1
    lines.append(
        f"{substrate_index}  {substrate.material}  {substrate.thickness_nm:.6g}  "
        f"{Ns.real:.6g}  {Ns.imag:.6g}"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def make_txt_best_checkpoint_saver(
    out_dir: str,
    *,
    incident: StackRow,
    substrate: StackRow,
    nk: dict[str, complex],
    film_indices: list[int],
    method: str,
    enabled: bool = True,
    plot_rt_live: bool = True,
    calc=None,
    plot_wls: list[float] | None = None,
    bands: list | None = None,
    theta0: float = 0.0,
    polarization: str = "unpolarized",
    nk_note: str = "",
):
    """Live-update ``stack_best.txt`` (+ CSV log / RT plot) on best improve."""
    if not enabled:
        return None

    os.makedirs(out_dir, exist_ok=True)
    stack_live = os.path.join(out_dir, "stack_best.txt")
    log_path = os.path.join(out_dir, "best_updates.csv")
    state = {"n": 0}
    if not os.path.isfile(log_path):
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(
                "update,stage,iter,n_eval,n_improve,cost,delta,thickness_delta_nm,"
                "total_thickness_nm\n"
            )

    def on_best(layers, cost, info) -> None:
        state["n"] += 1
        total_nm = sum(d for _, d in layers) / _NM
        delta = float(info.get("delta", 0.0) or 0.0)
        d_rms = float(info.get("thickness_delta_nm", 0.0) or 0.0)
        headers = [
            f"# live best  method={method}  update=#{state['n']}  "
            f"stage={info.get('stage', '')}  "
            f"loss={float(cost):.12e}  Δ={delta:+.6e}  "
            f"d_rms={d_rms:.3f} nm  "
            f"iter={info.get('iter', 0)}  n_eval={info.get('n_eval', 0)}",
            "# rewritten whenever the running best improves (cost + Δ)",
        ]
        if nk_note:
            headers.insert(1, nk_note)
        write_stack_txt(
            stack_live,
            incident,
            layers,
            substrate,
            nk,
            film_indices=film_indices,
            header_lines=headers,
        )
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(
                f"{state['n']},"
                f"{info.get('stage', '')},"
                f"{info.get('iter', 0)},"
                f"{info.get('n_eval', 0)},"
                f"{info.get('n_improve', 0)},"
                f"{float(cost):.12e},"
                f"{delta:.12e},"
                f"{d_rms:.6f},"
                f"{total_nm:.4f}\n"
            )
        if (
            plot_rt_live
            and calc is not None
            and plot_wls
            and bands is not None
        ):
            R, T = calc.spectrum(
                layers,
                plot_wls,
                theta0,
                incident=incident.material,
                substrate=substrate.material,
                polarization=polarization,
            )
            plot_rt(
                os.path.join(out_dir, "rt_best.png"),
                plot_wls,
                R,
                T,
                bands=bands,
                title=(
                    f"live best  loss={float(cost):.4e}  "
                    f"Δ={delta:+.3e}  #{state['n']}"
                ),
                layers=list(layers),
            )
            write_spectrum_csv(
                os.path.join(out_dir, "spectrum_best.csv"), plot_wls, R, T
            )
            write_band_stats_csv(
                os.path.join(out_dir, "band_stats_best.csv"),
                plot_wls,
                R,
                T,
                bands,
            )
        print(
            f"    checkpoint: updated {stack_live}  "
            f"cost={float(cost):.6e}  Δ={delta:+.3e}  "
            f"d_rms={d_rms:.2f} nm  stage={info.get('stage', '')}  "
            f"update=#{state['n']}",
            flush=True,
        )

    return on_best


def write_loss_history(path: str, history: list[float], best_iter: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("iter,loss,is_best\n")
        for i, loss in enumerate(history):
            flag = 1 if i == best_iter else 0
            fh.write(f"{i},{loss:.12e},{flag}\n")


def write_loss_history_epochs(
    path: str, history: list[float], best_epoch: int
) -> None:
    """Same as write_loss_history but header uses epoch (mini-batch mode)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("epoch,loss,is_best\n")
        for i, loss in enumerate(history):
            flag = 1 if i == best_epoch else 0
            fh.write(f"{i},{loss:.12e},{flag}\n")


def print_band_report(
    label: str,
    layers: list[tuple[str, float]],
    rbands: list[RObjectiveBand],
    wavelengths: list[float],
    R: list[float],
) -> None:
    total = sum(d for _, d in layers) / _NM
    print(f"\n{label}")
    print(f"  {len(layers)} layers, total thin-film thickness {total:.1f} nm")
    for i, (m, d) in enumerate(layers, 1):
        print(f"    {i:2d}. {m:<8} {d / _NM:8.2f} nm")
    for b in rbands:
        rs = [
            r
            for wl, r in zip(wavelengths, R)
            if b.wl_lo - 1e-15 <= wl <= b.wl_hi + 1e-15
        ]
        if not rs:
            continue
        goal = "high-R" if b.maximize else "low-R"
        print(
            f"  {b.wl_lo / _NM:.0f}–{b.wl_hi / _NM:.0f} nm  ({goal}, "
            f"R_target={b.r_target:g}, w={b.weight:g}): "
            f"R mean/min/max = {100 * sum(rs)/len(rs):.1f}/"
            f"{100 * min(rs):.1f}/{100 * max(rs):.1f}%"
        )


def run(stack_path: str, cfg_path: str) -> int:
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)

    incident, film_rows, substrate = load_stack_txt(stack_path)
    nk_source = parse_nk_source(cfg)
    ref_nm = float(cfg.get("nk_ref_wavelength_nm", _NK_REF_NM_DEFAULT))
    ref_wl = ref_nm * _NM

    material_names = [
        incident.material,
        *[r.material for r in film_rows],
        substrate.material,
    ]
    if nk_source == "library":
        require_library_materials(*material_names)
        nk = library_nk_table(material_names, ref_wavelength_m=ref_wl)
    else:
        nk = nk_table(incident, film_rows, substrate)

    # Only coating layers are free; incident + substrate stay fixed.
    layers_file = [(r.material, r.thickness_m) for r in film_rows]
    rbands = parse_r_bands(cfg)
    layers0, init_note = apply_init_layers(layers_file, rbands, cfg)
    film_indices = [r.index for r in film_rows]
    free_indices = list(range(len(layers0)))
    use_needle = bool(cfg.get("use_needle", False))

    method = str(cfg.get("method", "auto")).lower()
    multistart_method = str(cfg.get("multistart_method", "trf")).lower()
    angle_deg = float(cfg.get("incident_angle_deg", 0.0))
    theta0 = math.radians(angle_deg)
    pol = str(cfg.get("polarization", "unpolarized"))
    out_dir = cfg.get(
        "output_dir",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "optimize"),
    )
    if not os.path.isabs(out_dir):
        out_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(stack_path)), out_dir)
        )

    use_cuda = bool(cfg.get("use_cuda", False))
    raw_gpu_ids = cfg.get("multistart_gpu_ids")
    if raw_gpu_ids is None:
        multistart_gpu_ids: list[int] = []
    else:
        if not isinstance(raw_gpu_ids, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_gpu_ids
        ):
            raise ValueError("multistart_gpu_ids must be a JSON array of GPU IDs")
        multistart_gpu_ids = list(raw_gpu_ids)
    if multistart_gpu_ids and not use_cuda:
        raise ValueError("multistart_gpu_ids requires use_cuda=true")
    if multistart_gpu_ids:
        import tmm_cuda

        cp = tmm_cuda.require_cupy()
        device_count = int(cp.cuda.runtime.getDeviceCount())
        invalid_ids = [
            device for device in multistart_gpu_ids
            if device < 0 or device >= device_count
        ]
        if invalid_ids:
            raise ValueError(
                f"invalid CUDA device IDs {invalid_ids}; "
                f"available IDs are 0..{device_count - 1}"
            )
    if nk_source == "fixed":
        calc = ConstantNkCalculator(nk, use_cuda=use_cuda)
    else:
        calc = make_calculator(use_cuda=use_cuda)

    # Mini-batch: keep full-grid Adam by default; enable via mini_batch=true
    # or a nested object {"batch_size", "n_batches", ...}.
    mb_raw = cfg.get("mini_batch", False)
    if isinstance(mb_raw, dict):
        mini_batch = True
        mb_cfg = mb_raw
    else:
        mini_batch = bool(mb_raw)
        mb_cfg = cfg
    batch_size = int(mb_cfg.get("batch_size", cfg.get("batch_size", 8)))
    n_batches = mb_cfg.get("n_batches", cfg.get("n_batches"))
    if n_batches is not None:
        n_batches = int(n_batches)
    n_epochs = mb_cfg.get("n_epochs", cfg.get("n_epochs"))
    if n_epochs is not None:
        n_epochs = int(n_epochs)
    shuffle_seed = mb_cfg.get("shuffle_seed", cfg.get("shuffle_seed"))
    if shuffle_seed is not None:
        shuffle_seed = int(shuffle_seed)

    opt = MaxMinROptimizer(
        calc,
        rbands,
        theta0=theta0,
        incident=incident.material,
        substrate=substrate.material,
        polarization=pol,
        substrate_model="semi_infinite",
        wavelength_step=_NM * float(cfg.get("wavelength_step_nm", 10)),
        thickness_weight=float(cfg.get("thickness_weight", 0.0)),
        fd_step=_NM * float(cfg.get("fd_step_nm", 0.5)),
        lambda0=float(cfg.get("lambda0", 1e-2)),
        max_iter=int(cfg.get("max_iter", 40)),
        method=method,
        adam_lr=_NM * float(cfg.get("adam_lr_nm", 2.0)),
        adam_beta1=float(cfg.get("adam_beta1", 0.9)),
        adam_beta2=float(cfg.get("adam_beta2", 0.999)),
        adam_eps=float(cfg.get("adam_eps", 1e-8)),
        adam_max_step=_NM * float(cfg.get("adam_max_step_nm", 10.0)),
        cg_initial_step=_NM
        * float(cfg.get("cg_initial_step_nm", cfg.get("adam_lr_nm", 2.0))),
        cg_max_step=_NM
        * float(cfg.get("cg_max_step_nm", cfg.get("adam_max_step_nm", 10.0))),
        cg_restart=(
            int(cfg["cg_restart"]) if cfg.get("cg_restart") is not None else None
        ),
        lbfgs_m=int(cfg.get("lbfgs_m", 10)),
        lbfgs_maxls=int(cfg.get("lbfgs_maxls", 20)),
        trf_x_scale=cfg.get("trf_x_scale", "jac"),
        multistart_n=int(cfg.get("multistart_n", 8)),
        multistart_method=multistart_method,
        multistart_seed=(
            int(cfg["multistart_seed"])
            if cfg.get("multistart_seed") is not None
            else None
        ),
        multistart_sampler=str(cfg.get("multistart_sampler", "lhs")),
        multistart_candidate_n=int(cfg.get("multistart_candidate_n", 256)),
        surrogate_trees=int(cfg.get("surrogate_trees", 200)),
        surrogate_exploration_beta=float(
            cfg.get("surrogate_exploration_beta", 1.0)
        ),
        surrogate_pool_n=int(cfg.get("surrogate_pool_n", 10000)),
        surrogate_diversity_weight=float(
            cfg.get("surrogate_diversity_weight", 0.1)
        ),
        multistart_final_polish_method=cfg.get(
            "multistart_final_polish_method"
        ),
        multistart_gpu_ids=multistart_gpu_ids,
        multistart_progress_interval_s=float(
            cfg.get("multistart_progress_interval_s", 10.0)
        ),
        multistart_sampling_bounds=parse_multistart_sampling_bounds_nm(
            cfg.get("multistart_sampling_bounds_nm")
        ),
        auto_de_fallback=bool(cfg.get("auto_de_fallback", True)),
        auto_min_relative_improvement=float(
            cfg.get("auto_min_relative_improvement", 0.01)
        ),
        min_thickness=_NM * float(cfg.get("min_thickness_nm", 8.0)),
        max_total_thickness=(
            _NM * float(cfg["max_total_thickness_nm"])
            if cfg.get("max_total_thickness_nm") is not None
            else None
        ),
        thickness_bounds=parse_thickness_bounds_nm(
            cfg.get("thickness_bounds_nm")
        ),
        mini_batch=mini_batch
        and (
            method == "adam"
            or (
                method in ("auto", "hybrid", "multistart", "multi_start", "multi-start")
                and multistart_method == "adam"
            )
        ),
        batch_size=batch_size,
        n_batches=n_batches,
        n_epochs=n_epochs,
        shuffle_seed=shuffle_seed,
        de_popsize=int(cfg.get("de_popsize", 15)),
        de_mutation=(
            tuple(cfg["de_mutation"])
            if isinstance(cfg.get("de_mutation"), list)
            else cfg.get("de_mutation", (0.5, 1.0))
        ),
        de_recombination=float(cfg.get("de_recombination", 0.7)),
        global_seed=(
            int(cfg["global_seed"]) if cfg.get("global_seed") is not None else None
        ),
        global_polish=bool(cfg.get("global_polish", True)),
        global_polish_lm=bool(cfg.get("global_polish_lm", False)),
        global_polish_method=cfg.get("global_polish_method"),
        da_initial_temp=float(cfg.get("da_initial_temp", 5230.0)),
        da_visit=float(cfg.get("da_visit", 2.62)),
        da_accept=float(cfg.get("da_accept", -5.0)),
        checkpoint_local_every=cfg.get("checkpoint_local_every"),
    )

    os.makedirs(out_dir, exist_ok=True)
    checkpoint_on_best = bool(cfg.get("checkpoint_on_best", True))
    checkpoint_plot_rt = bool(cfg.get("checkpoint_plot_rt", True))

    print("Text-stack R-target least-squares optimizer")
    print(f"  stack: {stack_path}")
    print(f"  config: {cfg_path}")
    print(f"  method: {method}  angle: {angle_deg:g} deg  pol: {pol}")
    if method in ("auto", "hybrid", "multistart", "multi_start", "multi-start"):
        print(
            f"  multistart_sampler: {opt.multistart_sampler}  "
            f"n={opt.multistart_n}"
        )
    print(f"  min_thickness_nm: {opt.min_thickness / _NM:g}")
    if opt.max_total_thickness is not None:
        print(
            f"  max_total_thickness_nm: "
            f"{opt.max_total_thickness / _NM:g} (hard constraint)"
        )
    if nk_source == "library":
        print(
            f"  nk_source: library  (dispersion DB; "
            f"stack n,k columns ignored; ref export @ {ref_nm:g} nm)"
        )
    else:
        print("  nk_source: fixed  (constant n,k from stack file)")
    if use_cuda:
        print("  use_cuda: True (CuPy wavelength-batched TMM)")
    if multistart_gpu_ids:
        print(f"  multistart_gpu_ids: {multistart_gpu_ids}")
    if checkpoint_on_best:
        print(f"  checkpoint_on_best: {os.path.join(out_dir, 'stack_best.txt')}")
    if opt.mini_batch:
        print(
            f"  mini-batch: batch_size={opt.batch_size}  "
            f"n_batches={opt.n_batches}  n_epochs={opt.n_epochs}  "
            f"seed={opt.shuffle_seed}"
        )
    else:
        print(f"  training: full wavelength grid  max_iter={opt.max_iter}")
    print(
        f"  fixed incident: {incident.material}  "
        f"d={incident.thickness_nm:g} nm  "
        f"n={nk[dsp.normalize_material_name(incident.material)].real:g}  "
        f"k={nk[dsp.normalize_material_name(incident.material)].imag:g}"
        + (f"  (@{ref_nm:g} nm)" if nk_source == "library" else "")
    )
    print(
        f"  fixed substrate: {substrate.material}  "
        f"d={substrate.thickness_nm:g} nm  "
        f"n={nk[dsp.normalize_material_name(substrate.material)].real:g}  "
        f"k={nk[dsp.normalize_material_name(substrate.material)].imag:g}"
        + (f"  (@{ref_nm:g} nm)" if nk_source == "library" else "")
    )
    print(
        f"  free coating layers: {len(layers0)}  "
        f"(incident/substrate thicknesses not optimised)"
    )
    print(f"  init: {init_note}")
    if init_note.startswith("random"):
        print("  init thicknesses (nm):")
        for i, (m, d) in enumerate(layers0, 1):
            print(f"    {i:2d}. {m:<10} {d / _NM:8.2f}")
    print(
        f"  objective: band-normalized MSE(R→target) "
        f"+ thickness_weight={opt.thickness_weight:g}"
    )
    print(f"  n_bands: {len(rbands)}")
    for b in rbands:
        print(
            f"    {b.wl_lo / _NM:.0f}–{b.wl_hi / _NM:.0f} nm  "
            f"R_target={b.r_target:g}  weight={b.weight:g}"
        )

    plot_lo, plot_hi = cfg.get("plot_wavelength_nm", [400, 1800])
    plot_step = float(cfg.get("plot_step_nm", 5))
    plot_wls = [x * _NM for x in dense_grid_nm(float(plot_lo), float(plot_hi), plot_step)]
    plot_bands = [b.as_band_spec() for b in rbands]

    if nk_source == "library":
        nk_note = (
            f"# nk_source=library  (n,k columns = dispersion DB @ {ref_nm:g} nm, "
            "for reference; TMM uses full dispersion)"
        )
    else:
        nk_note = "# nk_source=fixed  (constant n,k from input stack)"

    opt.on_best = make_txt_best_checkpoint_saver(
        out_dir,
        incident=incident,
        substrate=substrate,
        nk=nk,
        film_indices=None if use_needle else film_indices,
        method=method,
        enabled=checkpoint_on_best,
        plot_rt_live=checkpoint_plot_rt,
        calc=calc,
        plot_wls=plot_wls,
        bands=plot_bands,
        theta0=theta0,
        polarization=pol,
        nk_note=nk_note,
    )

    R0, T0 = calc.spectrum(
        layers0,
        plot_wls,
        theta0,
        incident=incident.material,
        substrate=substrate.material,
        polarization=pol,
    )
    print_band_report("Before", layers0, rbands, opt.wavelengths, opt._rt(layers0)[0])

    if use_needle:
        synthesizer = NeedleSynthesizer(
            opt,
            high_index=str(cfg.get("high_index", "tio2")),
            low_index=str(cfg.get("low_index", "sio2")),
            max_layers=int(cfg.get("max_layers", 32)),
            max_add_rounds=int(cfg.get("max_add_rounds", 12)),
            refine_after_add=bool(cfg.get("refine_after_add", True)),
            n_design_centres=int(cfg.get("n_design_centres", 8)),
            add_pair=bool(cfg.get("add_pair", False)),
            candidate_mode=str(cfg.get("needle_candidate_mode", "sensitivity")),
            deep_search_candidates=int(cfg.get("deep_search_candidates", 3)),
            needle_probe=_NM * float(cfg.get("needle_probe_nm", 1.0)),
            prune_threshold=_NM * float(cfg.get("prune_threshold_nm", 10.0)),
            ar_layers=int(cfg.get("ar_layers", 4)),
            ar_add_rounds=int(cfg.get("ar_add_rounds", 4)),
            front_free_extra=int(cfg.get("front_free_extra", 6)),
        )
        synthesis = synthesizer.run(layers0, verbose=True)
        result = synthesis.lm
        result.layers = list(synthesis.layers)
        result.cost = opt.cost(result.layers)
        result.residuals = opt.residuals(result.layers)
        result.final_layers = list(result.layers)
        result.final_cost = result.cost
        result.message = f"needle({synthesis.n_added} added)+{result.message}"
    else:
        result = opt.optimize(layers0, free_indices=free_indices, verbose=True)
    result_film_indices = (
        film_indices if len(result.layers) == len(film_indices) else None
    )
    # The selected optimum is the checkpoint with the lowest loss.
    layers_best = result.layers
    layers_final = result.final_layers if result.final_layers is not None else layers_best
    # Every method reports and optimizes the same residual least-squares merit.
    best_cost = opt.cost(layers_best)
    final_cost = opt.cost(layers_final)
    best_iter = result.best_iter
    start_cost = opt.cost(layers0)
    best_delta = best_cost - start_cost
    final_delta = final_cost - start_cost

    R1, T1 = calc.spectrum(
        layers_best,
        plot_wls,
        theta0,
        incident=incident.material,
        substrate=substrate.material,
        polarization=pol,
    )
    Rf, Tf = calc.spectrum(
        layers_final,
        plot_wls,
        theta0,
        incident=incident.material,
        substrate=substrate.material,
        polarization=pol,
    )
    if result.pre_polish is not None:
        g = result.pre_polish
        print_band_report(
            f"After global ({g.message})",
            g.layers,
            rbands,
            opt.wavelengths,
            opt._rt(g.layers)[0],
        )
        print(f"  global loss={g.cost:.6e}", flush=True)
    print_band_report(
        f"Best (lowest loss, {'epoch' if opt.mini_batch else 'iter'} {best_iter})",
        layers_best,
        rbands,
        opt.wavelengths,
        opt._rt(layers_best)[0],
    )
    print(
        f"\n  best loss={best_cost:.6e}  Δ={best_delta:+.3e} at "
        f"{'epoch' if opt.mini_batch else 'iter'} {best_iter}  "
        f"(ran {result.n_iter} "
        f"{'epochs' if opt.mini_batch else 'iters'}, {result.message})"
    )
    if layers_final is not layers_best:
        print_band_report(
            "Final iterate",
            layers_final,
            rbands,
            opt.wavelengths,
            opt._rt(layers_final)[0],
        )
        print(f"  final loss={final_cost:.6e}  Δ={final_delta:+.3e}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "spectrum_before_after.csv")
    plot_path = os.path.join(out_dir, "rt_before_after.png")
    stack_best = os.path.join(out_dir, "stack_best.txt")
    stack_final = os.path.join(out_dir, "stack_final.txt")
    loss_path = os.path.join(out_dir, "loss_history.csv")
    # nk_note already set before optimize for live checkpoints.

    # Optical constants actually used by TMM over the plot grid.
    used_mats = materials_used_in_stack(
        incident.material, layers_best, substrate.material
    )
    nk_curves = sample_nk_curves(
        used_mats,
        plot_wls,
        fixed_nk=nk if nk_source == "fixed" else None,
    )
    nk_label = nk_source

    # CSV with before/after columns (after = best-loss stack).
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("wavelength_nm,R_before,T_before,R_after,T_after\n")
        for wl, rb, tb, ra, ta in zip(plot_wls, R0, T0, R1, T1):
            fh.write(
                f"{wl / _NM:.2f},{rb:.6f},{tb:.6f},{ra:.6f},{ta:.6f}\n"
            )

    plot_results(
        plot_path,
        plot_wls,
        R0,
        T0,
        R1,
        T1,
        plot_bands,
        materials_to_show=[],
        layers_before=layers0,
        layers_after=layers_best,
        nk_by_material=nk_curves,
        nk_source_label=nk_label,
    )
    plot_used_nk(
        os.path.join(out_dir, "nk_used.png"),
        plot_wls,
        nk_curves,
        source_label=nk_label,
    )
    write_nk_csv(os.path.join(out_dir, "nk_used.csv"), plot_wls, nk_curves)
    plot_rt(
        os.path.join(out_dir, "rt_best.png"),
        plot_wls,
        R1,
        T1,
        bands=plot_bands,
        title=(
            f"{os.path.basename(stack_path)} best "
            f"({method.upper()} iter {best_iter}, "
            f"loss={best_cost:.4e}, Δ={best_delta:+.3e})"
        ),
        layers=layers_best,
    )
    plot_rt(
        os.path.join(out_dir, "rt_final.png"),
        plot_wls,
        Rf,
        Tf,
        bands=plot_bands,
        title=(
            f"{os.path.basename(stack_path)} final "
            f"({method.upper()}, loss={final_cost:.4e}, Δ={final_delta:+.3e})"
        ),
        layers=layers_final,
    )
    write_stack_txt(
        stack_best,
        incident,
        layers_best,
        substrate,
        nk,
        film_indices=result_film_indices,
        header_lines=[
            f"# best stack (lowest loss)  method={method}  "
            f"loss={best_cost:.12e}  Δ={best_delta:+.6e}  "
            f"best_iter={best_iter}  "
            f"n_iter={result.n_iter}  msg={result.message}",
            nk_note,
            "# incident and substrate thicknesses fixed (not optimised)",
            "# R_target: per-band reflectance target in [0, 1]",
        ],
    )
    write_stack_txt(
        stack_final,
        incident,
        layers_final,
        substrate,
        nk,
        film_indices=result_film_indices,
        header_lines=[
            f"# final iterate  method={method}  "
            f"loss={final_cost:.12e}  Δ={final_delta:+.6e}  "
            f"n_iter={result.n_iter}  msg={result.message}",
            nk_note,
            "# last iterate (may differ from stack_best.txt)",
        ],
    )
    # Alias for callers expecting the previous filename.
    write_stack_txt(
        os.path.join(out_dir, "stack_optimised.txt"),
        incident,
        layers_best,
        substrate,
        nk,
        film_indices=result_film_indices,
        header_lines=[
            f"# best stack (same as stack_best.txt)  "
            f"loss={best_cost:.12e}  best_iter={best_iter}",
            nk_note,
            "# incident and substrate thicknesses fixed (not optimised)",
        ],
    )
    if result.pre_polish is not None:
        g = result.pre_polish
        Rg, Tg = calc.spectrum(
            g.layers,
            plot_wls,
            theta0,
            incident=incident.material,
            substrate=substrate.material,
            polarization=pol,
        )
        write_stack_txt(
            os.path.join(out_dir, "stack_global.txt"),
            incident,
            g.layers,
            substrate,
            nk,
            film_indices=result_film_indices,
            header_lines=[
                f"# global-stage stack  method={method}  "
                f"loss={g.cost:.12e}  msg={g.message}",
                nk_note,
                "# before local polish (lm/adam)",
            ],
        )
        write_stack_txt(
            os.path.join(out_dir, "stack_polished.txt"),
            incident,
            layers_best,
            substrate,
            nk,
            film_indices=result_film_indices,
            header_lines=[
                f"# polished stack  method={result.message}  "
                f"loss={best_cost:.12e}  global_loss={g.cost:.12e}",
                nk_note,
                "# after local polish",
            ],
        )
        with open(os.path.join(out_dir, "spectrum_global.csv"), "w", encoding="utf-8") as fh:
            fh.write(
                "wavelength_nm,R_before,T_before,R_global,T_global,"
                "R_polished,T_polished\n"
            )
            for wl, rb, tb, rg, tg, ra, ta in zip(
                plot_wls, R0, T0, Rg, Tg, R1, T1
            ):
                fh.write(
                    f"{wl / _NM:.2f},{rb:.6f},{tb:.6f},"
                    f"{rg:.6f},{tg:.6f},{ra:.6f},{ta:.6f}\n"
                )
        print(f"  wrote {os.path.join(out_dir, 'stack_global.txt')}")
        print(f"  wrote {os.path.join(out_dir, 'stack_polished.txt')}")
        print(f"  wrote {os.path.join(out_dir, 'spectrum_global.csv')}")
    if opt.mini_batch:
        write_loss_history_epochs(loss_path, result.history, best_iter)
    else:
        write_loss_history(loss_path, result.history, best_iter)
    write_spectrum_csv(
        os.path.join(out_dir, "spectrum_best.csv"), plot_wls, R1, T1
    )
    write_spectrum_csv(
        os.path.join(out_dir, "spectrum_final.csv"), plot_wls, Rf, Tf
    )
    write_band_stats_csv(
        os.path.join(out_dir, "band_stats_best.csv"),
        plot_wls,
        R1,
        T1,
        plot_bands,
    )
    write_band_stats_csv(
        os.path.join(out_dir, "band_stats_final.csv"),
        plot_wls,
        Rf,
        Tf,
        plot_bands,
    )

    print(f"\n  wrote {csv_path}")
    print(f"  wrote {stack_best}")
    print(f"  wrote {stack_final}")
    print(f"  wrote {loss_path}")
    print(f"  wrote {plot_path}")
    print(f"  wrote {os.path.join(out_dir, 'nk_used.png')}")
    print(f"  wrote {os.path.join(out_dir, 'nk_used.csv')}")
    print(f"  wrote {os.path.join(out_dir, 'rt_best.png')}")
    print(f"  wrote {os.path.join(out_dir, 'rt_final.png')}")
    print(f"  wrote {os.path.join(out_dir, 'band_stats_best.csv')}")
    print(f"  wrote {os.path.join(out_dir, 'band_stats_final.csv')}")
    close_all_figures()
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_matplotlib()
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Bounded thickness optimization and optional Needle "
        "synthesis for joint reflectance targets."
    )
    ap.add_argument(
        "stack",
        nargs="?",
        default=os.path.join(here, "examples", "example_stack.txt"),
        help="initial stack text file (plot_rt_txt format)",
    )
    ap.add_argument(
        "config",
        nargs="?",
        default=os.path.join(here, "examples", "example_optimize_film.json"),
        help="JSON with bands[].R_target / objective; "
        "method=auto|multistart|trf|adam|lm|cg|lbfgs|de|dual_annealing",
    )
    args = ap.parse_args(argv)
    return run(args.stack, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
