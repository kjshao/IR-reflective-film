"""Thickness-only Adam/LM optimiser for a plain-text constant-n,k stack.

Initial stack format matches ``plot_rt_txt.py``. Optimises **coating** layer
thicknesses only; the first row (incident medium) and last row (substrate)
are never free variables — their thicknesses stay exactly as in the input.

Objective: build a piecewise R target (minimize bands → 0, maximize → 1),
then minimise the **band-normalized** fit

    L = Σ_b w_b · mean_{λ∈b} |R − t_b|^p  /  Σ_b w_b

(independent of sample count and absolute weight scale), plus optional
peak-suppression terms (``smooth_weight``, ``ripple_weight``; raise
``error_power`` above 2 to overweight sharp outliers).

Usage::

    python3 sim/optimize_rt_txt.py \\
        sim/examples/example_stack.txt \\
        sim/examples/example_optimize_rt_txt.json

Training modes (``method=adam``):

  - **full grid** (default): each iteration uses the full wavelength grid.
  - **mini-batch**: set ``mini_batch`` true or to an object with
    ``batch_size``, ``n_batches``, optional ``n_epochs`` / ``shuffle_seed``.
    Each epoch draws ``n_batches`` batches; every batch picks a random float
    start in ``[λ_min, λ_max)`` and takes ``batch_size`` points spaced by
    ``(λ_max−λ_min)/batch_size``, wrapping at ``λ_max`` back to ``λ_min``.
    One Adam step per batch; full-grid cost is recorded per epoch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tmm
from lm_optimizer import BandSpec, LMThicknessOptimizer
from optimize_film import dense_grid_nm, plot_results
from plot_rt import plot_rt, write_spectrum_csv
from plot_rt_txt import StackRow, load_stack_txt
from rt_calculator import DEFAULT_SUBSTRATE_THICKNESS, RTCalculator

_NM = 1e-9


@dataclass
class RObjectiveBand:
    """One wavelength interval with a reflectance max or min objective."""

    wl_lo: float
    wl_hi: float
    maximize: bool
    weight: float = 1.0

    def as_band_spec(self) -> BandSpec:
        """Map to BandSpec for grid / shaded plot targets."""
        if self.maximize:
            return BandSpec(
                wl_lo=self.wl_lo,
                wl_hi=self.wl_hi,
                R_min=0.9,
                weight=self.weight,
                R_target=1.0,
            )
        return BandSpec(
            wl_lo=self.wl_lo,
            wl_hi=self.wl_hi,
            R_max=0.1,
            weight=self.weight,
            R_target=0.0,
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
        if obj is None:
            raise ValueError(
                f"bands[{i}]: need objective maximize|minimize "
                "(aliases: R/goal max|min)"
            )
        out.append(
            RObjectiveBand(
                wl_lo=lo * _NM,
                wl_hi=hi * _NM,
                maximize=_parse_objective(obj),
                weight=float(b.get("weight", 1.0)),
            )
        )
    return out


class ConstantNkCalculator(RTCalculator):
    """TMM using fixed N=n+ik from the text stack (no dispersion library)."""

    def __init__(self, nk: dict[str, complex]):
        self.nk = {name.lower(): complex(val) for name, val in nk.items()}

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
    """Piecewise R target: maximize→1, minimize→0; None outside all bands.

    If bands overlap, the last listed band wins for that wavelength.
    """
    targets: list[float | None] = [None] * len(wavelengths)
    weights = [0.0] * len(wavelengths)
    for b in bands:
        t = 1.0 if b.maximize else 0.0
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


def reflectance_mse(
    R: Sequence[float],
    bands: Sequence[RObjectiveBand],
    wavelengths: Sequence[float],
    *,
    error_power: float = 2.0,
) -> float:
    """Band-normalized fit loss, independent of N and absolute weight scale.

    For each band ``b`` compute the in-band mean ``mean_b |R - t_b|^p``, then

        L = Σ_b w_b · mean_b / Σ_b w_b

    So widening a band (more sample points) or scaling all ``w_b`` by a
    constant leaves ``L`` unchanged; only *relative* band weights matter.
    """
    p = float(error_power)
    if p <= 0:
        raise ValueError(f"error_power must be > 0, got {error_power}")
    band_indices = _band_sample_indices(bands, wavelengths)
    w_sum = _active_band_weight_sum(bands, band_indices)
    if w_sum <= 0.0:
        return 0.0
    loss = 0.0
    for b, idx in zip(bands, band_indices):
        w = max(float(b.weight), 0.0)
        if not idx or w <= 0.0:
            continue
        t = 1.0 if b.maximize else 0.0
        mean_err = sum(abs(R[i] - t) ** p for i in idx) / len(idx)
        loss += (w / w_sum) * mean_err
    return loss


def peak_suppression_penalty(
    R: Sequence[float],
    bands: Sequence[RObjectiveBand],
    wavelengths: Sequence[float],
    *,
    smooth_weight: float,
    ripple_weight: float,
) -> float:
    """Band-normalized smoothness + ripple; same weight/N independence as fit."""
    band_indices = _band_sample_indices(bands, wavelengths)
    w_sum = _active_band_weight_sum(bands, band_indices)
    if w_sum <= 0.0:
        return 0.0

    pen = 0.0
    if smooth_weight > 0:
        for b, idx in zip(bands, band_indices):
            w = max(float(b.weight), 0.0)
            if not idx or w <= 0.0:
                continue
            diffs = [
                R[j] - R[i]
                for i, j in zip(idx, idx[1:])
                if j == i + 1
            ]
            if not diffs:
                continue
            mean_d2 = sum(d * d for d in diffs) / len(diffs)
            pen += smooth_weight * (w / w_sum) * mean_d2

    if ripple_weight > 0:
        for b, idx in zip(bands, band_indices):
            w = max(float(b.weight), 0.0)
            if len(idx) < 2 or w <= 0.0:
                continue
            rs = [R[i] for i in idx]
            amp = max(rs) - min(rs)
            pen += ripple_weight * (w / w_sum) * (amp * amp)

    return pen


def build_maxmin_r_residuals(
    layers: Sequence[tuple[str, float]],
    bands: Sequence[RObjectiveBand],
    wavelengths: Sequence[float],
    R: Sequence[float],
    *,
    thickness_weight: float = 0.0,
    thickness_ref: float = 1000e-9,
    error_power: float = 2.0,
    smooth_weight: float = 0.0,
    ripple_weight: float = 0.0,
) -> list[float]:
    """Residuals for band-normalized fit + peak-suppression regularisers.

    Fit term (independent of sample count and absolute weight scale)::

        L_fit = Σ_b w_b · mean_{λ∈b} |R − t_b|^p  /  Σ_b w_b

    with ``t_b = 1`` (maximize) or ``0`` (minimize). Residuals are scaled so
    ``0.5 * sum(r**2) == L_fit + L_smooth + L_ripple (+ thickness)``.

    Peak suppression::

      - ``error_power`` > 2 (e.g. 4): overweight large |R−target| outliers
      - ``smooth_weight``: band-normalized mean (ΔR)²
      - ``ripple_weight``: band-normalized (R_max − R_min)²
    """
    p = float(error_power)
    if p <= 0:
        raise ValueError(f"error_power must be > 0, got {error_power}")

    band_indices = _band_sample_indices(bands, wavelengths)
    w_sum = _active_band_weight_sum(bands, band_indices)
    if w_sum <= 0.0:
        return [0.0]

    res: list[float] = []
    for b, idx in zip(bands, band_indices):
        w = max(float(b.weight), 0.0)
        if not idx or w <= 0.0:
            continue
        t = 1.0 if b.maximize else 0.0
        n = len(idx)
        # 0.5 Σ_i r_i² = (w/W) · mean |e|^p
        scale = math.sqrt(2.0 * w / (w_sum * n))
        for i in idx:
            e = R[i] - t
            mag = abs(e) ** (p / 2.0)
            res.append(scale * (mag if e >= 0.0 else -mag))

    if smooth_weight > 0:
        for b, idx in zip(bands, band_indices):
            w = max(float(b.weight), 0.0)
            if not idx or w <= 0.0:
                continue
            pairs = [
                (i, j)
                for i, j in zip(idx, idx[1:])
                if j == i + 1
            ]
            if not pairs:
                continue
            scale = math.sqrt(2.0 * smooth_weight * w / (w_sum * len(pairs)))
            for i, j in pairs:
                res.append(scale * (R[j] - R[i]))

    if ripple_weight > 0:
        for b, idx in zip(bands, band_indices):
            w = max(float(b.weight), 0.0)
            if len(idx) < 2 or w <= 0.0:
                continue
            rs = [R[i] for i in idx]
            amp = max(rs) - min(rs)
            scale = math.sqrt(2.0 * ripple_weight * w / w_sum)
            res.append(scale * amp)

    if thickness_weight > 0 and layers:
        total = sum(d for _, d in layers)
        res.append(math.sqrt(2.0 * thickness_weight) * total / thickness_ref)
    return res


class MaxMinROptimizer(LMThicknessOptimizer):
    """Adam/LM thickness optimiser minimizing target MSE (+ peak penalties)."""

    def __init__(
        self,
        calculator: RTCalculator,
        rbands: Sequence[RObjectiveBand],
        *,
        error_power: float = 2.0,
        smooth_weight: float = 0.0,
        ripple_weight: float = 0.0,
        **kwargs,
    ):
        self.rbands = list(rbands)
        self.error_power = float(error_power)
        self.smooth_weight = float(smooth_weight)
        self.ripple_weight = float(ripple_weight)
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
            error_power=self.error_power,
            smooth_weight=self.smooth_weight,
            ripple_weight=self.ripple_weight,
        )

    def cost(self, layers: Sequence[tuple[str, float]]) -> float:
        """Band-normalized fit + peak terms (matches ``0.5*||r||^2``)."""
        R, _T = self._rt(layers)
        loss = reflectance_mse(
            R,
            self.rbands,
            self.wavelengths,
            error_power=self.error_power,
        )
        loss += peak_suppression_penalty(
            R,
            self.rbands,
            self.wavelengths,
            smooth_weight=self.smooth_weight,
            ripple_weight=self.ripple_weight,
        )
        if self.thickness_weight > 0 and layers:
            total = sum(d for _, d in layers)
            loss += self.thickness_weight * (total / 1000e-9) ** 2
        return loss


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
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines: list[str] = []
    if header_lines:
        lines.extend(header_lines)
    lines.append("# index  material  thickness_nm  n  k")
    lines.append(
        f"{incident.index}  {incident.material}  {incident.thickness_nm:.6g}  "
        f"{nk[incident.material.lower()].real:.6g}  "
        f"{nk[incident.material.lower()].imag:.6g}"
    )
    for i, (mat, d) in enumerate(films):
        idx = film_indices[i] if film_indices is not None else i + 1
        N = nk[mat.lower()]
        lines.append(
            f"{idx}  {mat}  {d / _NM:.4f}  {N.real:.6g}  {N.imag:.6g}"
        )
    lines.append(
        f"{substrate.index}  {substrate.material}  {substrate.thickness_nm:.6g}  "
        f"{nk[substrate.material.lower()].real:.6g}  "
        f"{nk[substrate.material.lower()].imag:.6g}"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_loss_history(path: str, history: list[float], best_iter: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("iter,mse,is_best\n")
        for i, loss in enumerate(history):
            flag = 1 if i == best_iter else 0
            fh.write(f"{i},{loss:.12e},{flag}\n")


def write_loss_history_epochs(
    path: str, history: list[float], best_epoch: int
) -> None:
    """Same as write_loss_history but header uses epoch (mini-batch mode)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("epoch,mse,is_best\n")
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
        goal = "maximize R" if b.maximize else "minimize R"
        print(
            f"  {b.wl_lo / _NM:.0f}–{b.wl_hi / _NM:.0f} nm  ({goal}, w={b.weight:g}): "
            f"R mean/min/max = {100 * sum(rs)/len(rs):.1f}/"
            f"{100 * min(rs):.1f}/{100 * max(rs):.1f}%"
        )


def run(stack_path: str, cfg_path: str) -> int:
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)

    incident, film_rows, substrate = load_stack_txt(stack_path)
    nk = nk_table(incident, film_rows, substrate)
    # Only coating layers are free; incident + substrate stay fixed.
    layers0 = [(r.material, r.thickness_m) for r in film_rows]
    film_indices = [r.index for r in film_rows]
    free_indices = list(range(len(layers0)))
    rbands = parse_r_bands(cfg)

    method = str(cfg.get("method", "adam")).lower()
    angle_deg = float(cfg.get("incident_angle_deg", 0.0))
    theta0 = math.radians(angle_deg)
    pol = str(cfg.get("polarization", "unpolarized"))
    out_dir = cfg.get(
        "output_dir",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "optimize_rt_txt"),
    )
    if not os.path.isabs(out_dir):
        out_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(stack_path)), out_dir)
        )

    calc = ConstantNkCalculator(nk)
    error_power = float(cfg.get("error_power", 2.0))
    smooth_weight = float(cfg.get("smooth_weight", 0.0))
    ripple_weight = float(cfg.get("ripple_weight", 0.0))

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
        error_power=error_power,
        smooth_weight=smooth_weight,
        ripple_weight=ripple_weight,
        fd_step=_NM * float(cfg.get("fd_step_nm", 0.5)),
        lambda0=float(cfg.get("lambda0", 1e-2)),
        max_iter=int(cfg.get("max_iter", 40)),
        method=method,
        adam_lr=_NM * float(cfg.get("adam_lr_nm", 2.0)),
        adam_beta1=float(cfg.get("adam_beta1", 0.9)),
        adam_beta2=float(cfg.get("adam_beta2", 0.999)),
        adam_eps=float(cfg.get("adam_eps", 1e-8)),
        adam_max_step=_NM * float(cfg.get("adam_max_step_nm", 10.0)),
        mini_batch=mini_batch and method == "adam",
        batch_size=batch_size,
        n_batches=n_batches,
        n_epochs=n_epochs,
        shuffle_seed=shuffle_seed,
    )

    print("Text-stack R-target MSE thickness optimiser")
    print(f"  stack: {stack_path}")
    print(f"  config: {cfg_path}")
    print(f"  method: {method}  angle: {angle_deg:g} deg  pol: {pol}")
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
        f"d={incident.thickness_nm:g} nm  n={incident.n:g}  k={incident.k:g}"
    )
    print(
        f"  fixed substrate: {substrate.material}  "
        f"d={substrate.thickness_nm:g} nm  n={substrate.n:g}  k={substrate.k:g}"
    )
    print(
        f"  free coating layers: {len(layers0)}  "
        f"(incident/substrate thicknesses not optimised)"
    )
    print(
        f"  objective: band-normalized mean(|R−target|^{error_power:g}) "
        f"+ smooth={smooth_weight:g} + ripple={ripple_weight:g} "
        f"(independent of N and absolute weights)"
    )
    print(f"  n_bands: {len(rbands)}")
    for b in rbands:
        goal = "maximize→1" if b.maximize else "minimize→0"
        print(
            f"    {b.wl_lo / _NM:.0f}–{b.wl_hi / _NM:.0f} nm  "
            f"R {goal}  weight={b.weight:g}"
        )

    plot_lo, plot_hi = cfg.get("plot_wavelength_nm", [400, 1800])
    plot_step = float(cfg.get("plot_step_nm", 5))
    plot_wls = [x * _NM for x in dense_grid_nm(float(plot_lo), float(plot_hi), plot_step)]

    R0, T0 = calc.spectrum(
        layers0,
        plot_wls,
        theta0,
        incident=incident.material,
        substrate=substrate.material,
        polarization=pol,
    )
    print_band_report("Before", layers0, rbands, opt.wavelengths, opt._rt(layers0)[0])

    result = opt.optimize(layers0, free_indices=free_indices, verbose=True)
    # Always keep the iterate with the lowest MSE seen during optimisation.
    layers_best = result.layers
    best_cost = result.cost
    best_iter = result.best_iter
    R1, T1 = calc.spectrum(
        layers_best,
        plot_wls,
        theta0,
        incident=incident.material,
        substrate=substrate.material,
        polarization=pol,
    )
    print_band_report(
        f"Best MSE ({'epoch' if opt.mini_batch else 'iter'} {best_iter})",
        layers_best,
        rbands,
        opt.wavelengths,
        opt._rt(layers_best)[0],
    )
    print(
        f"\n  best MSE={best_cost:.6e} at "
        f"{'epoch' if opt.mini_batch else 'iter'} {best_iter}  "
        f"(ran {result.n_iter} "
        f"{'epochs' if opt.mini_batch else 'iters'}, {result.message})"
    )

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "spectrum_before_after.csv")
    plot_path = os.path.join(out_dir, "rt_before_after.png")
    stack_best = os.path.join(out_dir, "stack_best.txt")
    loss_path = os.path.join(out_dir, "loss_history.csv")

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
        [b.as_band_spec() for b in rbands],
        materials_to_show=[],
    )
    plot_rt(
        os.path.join(out_dir, "rt_best.png"),
        plot_wls,
        R1,
        T1,
        bands=[b.as_band_spec() for b in rbands],
        title=(
            f"{os.path.basename(stack_path)} best MSE "
            f"({method.upper()} iter {best_iter})"
        ),
    )
    write_stack_txt(
        stack_best,
        incident,
        layers_best,
        substrate,
        nk,
        film_indices=film_indices,
        header_lines=[
            f"# best-MSE stack  method={method}  "
            f"mse={best_cost:.12e}  best_iter={best_iter}  "
            f"n_iter={result.n_iter}",
            "# incident and substrate thicknesses fixed (not optimised)",
            "# R_target: minimize bands → 0, maximize bands → 1",
        ],
    )
    # Alias for callers expecting the previous filename.
    write_stack_txt(
        os.path.join(out_dir, "stack_optimised.txt"),
        incident,
        layers_best,
        substrate,
        nk,
        film_indices=film_indices,
        header_lines=[
            f"# best-MSE stack (same as stack_best.txt)  "
            f"mse={best_cost:.12e}  best_iter={best_iter}",
            "# incident and substrate thicknesses fixed (not optimised)",
        ],
    )
    if opt.mini_batch:
        write_loss_history_epochs(loss_path, result.history, best_iter)
    else:
        write_loss_history(loss_path, result.history, best_iter)
    write_spectrum_csv(
        os.path.join(out_dir, "spectrum_best.csv"), plot_wls, R1, T1
    )

    print(f"\n  wrote {csv_path}")
    print(f"  wrote {stack_best}")
    print(f"  wrote {loss_path}")
    print(f"  wrote {plot_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Adam/LM thickness optimisation of a text-file stack "
        "for joint reflectance max/min over N wavelength bands."
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
        default=os.path.join(here, "examples", "example_optimize_rt_txt.json"),
        help="JSON with n_bands, bands[].objective, method=adam|lm",
    )
    args = ap.parse_args(argv)
    return run(args.stack, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
