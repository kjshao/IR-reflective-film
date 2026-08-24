"""Input-driven multilayer film design: TMM + LM + layer-by-layer synthesis.

Usage::

    python3 sim/optimize_film.py sim/examples/example_vis_pass_ir_reflect.json

Reads an input JSON describing angle, initial stack, band targets, and
optimiser options; writes optimised thicknesses, spectra CSV, and matplotlib
plots of reflectance / transmittance (and material refractive indices) before
and after optimisation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dispersion as dsp
from lm_optimizer import BandSpec, LMThicknessOptimizer
from needle import NeedleSynthesizer, make_optimizer_from_config
from rt_calculator import make_calculator, material_index

# Default materials available without extra files (also listed in dispersion.MATERIALS).
DEFAULT_MATERIALS = ("air", "sio2", "tio2", "glass")


def _nm(x: float) -> float:
    return x * 1e-9


def load_input(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    return cfg


def parse_bands(raw: list[dict]) -> list[BandSpec]:
    bands = []
    for b in raw:
        w = b["wavelength_nm"]
        if len(w) != 2:
            raise ValueError("each band needs wavelength_nm: [lo, hi]")
        bands.append(
            BandSpec(
                wl_lo=_nm(float(w[0])),
                wl_hi=_nm(float(w[1])),
                R_min=b.get("R_min"),
                R_max=b.get("R_max"),
                T_min=b.get("T_min"),
                T_max=b.get("T_max"),
                weight=float(b.get("weight", 1.0)),
                R_target=b.get("R_target"),
                T_target=b.get("T_target"),
            )
        )
    return bands


def parse_layers(raw: list[dict]) -> list[tuple[str, float]]:
    layers = []
    for layer in raw:
        mat = str(layer["material"]).lower()
        if "thickness_nm" in layer:
            d = _nm(float(layer["thickness_nm"]))
        elif "thickness_m" in layer:
            d = float(layer["thickness_m"])
        else:
            raise ValueError("layer needs thickness_nm or thickness_m")
        if mat not in dsp.MATERIALS:
            raise KeyError(
                f"material '{mat}' not in library; known: {sorted(dsp.MATERIALS)}"
            )
        layers.append((mat, d))
    return layers


def build_chirped_seed(seed: dict) -> list[tuple[str, float]]:
    """Quarter-wave chirped TiO2/SiO2 (or custom cell) starting stack."""
    centres = [_nm(float(c)) for c in seed.get("centres_nm", [900, 1200, 1500])]
    periods = int(seed.get("periods_per_centre", 3))
    cell = [str(m).lower() for m in seed.get("cell", ["tio2", "sio2"])]
    layers: list[tuple[str, float]] = []
    for lam0 in centres:
        for _ in range(periods):
            for mat in cell:
                if mat not in dsp.MATERIALS:
                    raise KeyError(f"seed material '{mat}' not in library")
                n = dsp.MATERIALS[mat](lam0).real
                layers.append((mat, 0.25 * lam0 / max(n, 1.01)))
    return layers


def resolve_initial_layers(cfg: dict) -> list[tuple[str, float]]:
    if cfg.get("layers"):
        return parse_layers(cfg["layers"])
    if cfg.get("seed"):
        return build_chirped_seed(cfg["seed"])
    # Minimal starter: one H/L pair near 1 µm.
    return build_chirped_seed({"centres_nm": [1000], "periods_per_centre": 1})


def plot_range_nm(cfg: dict, bands: list[BandSpec]) -> tuple[float, float]:
    if "plot_wavelength_nm" in cfg:
        lo, hi = cfg["plot_wavelength_nm"]
        return float(lo), float(hi)
    lo = min(b.wl_lo for b in bands) * 1e9
    hi = max(b.wl_hi for b in bands) * 1e9
    return lo, hi


def dense_grid_nm(lo_nm: float, hi_nm: float, step_nm: float) -> list[float]:
    n = max(1, int(round((hi_nm - lo_nm) / step_nm)))
    return [lo_nm + i * (hi_nm - lo_nm) / n for i in range(n + 1)]


def write_spectrum_csv(
    path: str,
    wavelengths_m: list[float],
    R_before: list[float],
    T_before: list[float],
    R_after: list[float],
    T_after: list[float],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("wavelength_nm,R_before,T_before,R_after,T_after\n")
        for wl, rb, tb, ra, ta in zip(
            wavelengths_m, R_before, T_before, R_after, T_after
        ):
            fh.write(
                f"{wl * 1e9:.2f},{rb:.6f},{tb:.6f},{ra:.6f},{ta:.6f}\n"
            )


def write_stack_json(path: str, layers: list[tuple[str, float]], meta: dict) -> None:
    payload = {
        **meta,
        "layers": [
            {"material": m, "thickness_nm": round(d * 1e9, 4)} for m, d in layers
        ],
        "total_thickness_nm": round(sum(d for _, d in layers) * 1e9, 4),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def plot_results(
    path: str,
    wavelengths_m: list[float],
    R_before: list[float],
    T_before: list[float],
    R_after: list[float],
    T_after: list[float],
    bands: list[BandSpec],
    materials_to_show: list[str] | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting; pip install matplotlib"
        ) from exc

    wl_nm = [w * 1e9 for w in wavelengths_m]
    materials_to_show = materials_to_show or ["sio2", "tio2", "glass"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    ax = axes[0]
    ax.plot(wl_nm, [100 * r for r in R_before], "--", color="C0", label="R before")
    ax.plot(wl_nm, [100 * r for r in R_after], "-", color="C0", label="R after")
    ax.plot(wl_nm, [100 * t for t in T_before], "--", color="C1", label="T before")
    ax.plot(wl_nm, [100 * t for t in T_after], "-", color="C1", label="T after")
    for b in bands:
        ax.axvspan(b.wl_lo * 1e9, b.wl_hi * 1e9, color="0.85", alpha=0.35, lw=0)
        if b.R_min is not None:
            ax.hlines(
                100 * b.R_min,
                b.wl_lo * 1e9,
                b.wl_hi * 1e9,
                colors="C0",
                linestyles=":",
                lw=1,
            )
        if b.R_max is not None:
            ax.hlines(
                100 * b.R_max,
                b.wl_lo * 1e9,
                b.wl_hi * 1e9,
                colors="C0",
                linestyles=":",
                lw=1,
            )
        if b.T_min is not None:
            ax.hlines(
                100 * b.T_min,
                b.wl_lo * 1e9,
                b.wl_hi * 1e9,
                colors="C1",
                linestyles=":",
                lw=1,
            )
        if b.T_max is not None:
            ax.hlines(
                100 * b.T_max,
                b.wl_lo * 1e9,
                b.wl_hi * 1e9,
                colors="C1",
                linestyles=":",
                lw=1,
            )
    ax.set_ylabel("R, T (%)")
    ax.set_ylim(-2, 105)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Reflectance & transmittance before / after optimisation")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(wl_nm, [100 * r for r in R_before], "--", label="R before")
    ax.plot(wl_nm, [100 * r for r in R_after], "-", label="R after")
    ax.set_ylabel("R (%)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Reflectance")

    ax = axes[2]
    for name in materials_to_show:
        if name not in dsp.MATERIALS:
            continue
        nk = material_index(name, wavelengths_m)
        ax.plot(wl_nm, [z.real for z in nk], label=f"n({name})")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("n")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Material refractive index (library)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_report(label: str, layers: list[tuple[str, float]], report: list[dict], ok: bool):
    total = sum(d for _, d in layers) * 1e9
    print(f"\n{label}")
    print(f"  {len(layers)} layers, total thin-film thickness {total:.1f} nm  "
          f"[{'SPECS MET' if ok else 'SPECS NOT MET'}]")
    for i, (m, d) in enumerate(layers, 1):
        print(f"    {i:2d}. {m:<8} {d * 1e9:8.2f} nm")
    for b in report:
        lo, hi = b["wl_nm"]
        tgt = b["targets"]
        bits = []
        if tgt["R_min"] is not None:
            bits.append(f"R≥{100*tgt['R_min']:.0f}%")
        if tgt["R_max"] is not None:
            bits.append(f"R≤{100*tgt['R_max']:.0f}%")
        if tgt["T_min"] is not None:
            bits.append(f"T≥{100*tgt['T_min']:.0f}%")
        if tgt["T_max"] is not None:
            bits.append(f"T≤{100*tgt['T_max']:.0f}%")
        print(
            f"  {lo:.0f}-{hi:.0f} nm  ({', '.join(bits) or 'no targets'}): "
            f"R[{100*b['R_min']:.1f}, {100*b['R_max']:.1f}]%  "
            f"T[{100*b['T_min']:.1f}, {100*b['T_max']:.1f}]%"
        )


def run(cfg: dict, input_path: str) -> int:
    bands = parse_bands(cfg["bands"])
    layers0 = resolve_initial_layers(cfg)
    angle_deg = float(cfg.get("incident_angle_deg", 0.0))
    theta0 = math.radians(angle_deg)
    opt_cfg = cfg.get("optimizer", {})
    out_dir = cfg.get(
        "output_dir",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "optimize"),
    )
    if not os.path.isabs(out_dir):
        # Resolve relative to the input file's directory for portability.
        out_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(input_path)), out_dir)
        )

    calc = make_calculator(
        cfg.get("rt_engine", "tmm"),
        cfg.get("external_command"),
    )

    substrate_model = cfg.get("substrate_model", "semi_infinite")
    lm_kwargs = {
        "theta0": theta0,
        "incident": cfg.get("incident_medium", "air"),
        "substrate": cfg.get("substrate", "glass"),
        "substrate_thickness": float(
            cfg.get("substrate_thickness_m", 0.7e-3)
        ),
        "exit_medium": cfg.get("exit_medium", "air"),
        "polarization": cfg.get("polarization", "unpolarized"),
        "substrate_model": substrate_model,
        "wavelength_step": _nm(float(opt_cfg.get("wavelength_step_nm", 10))),
        "thickness_weight": float(opt_cfg.get("thickness_weight", 0.02)),
        "fd_step": _nm(float(opt_cfg.get("fd_step_nm", 0.5))),
        "lambda0": float(opt_cfg.get("lambda0", 1e-2)),
        "max_iter": int(opt_cfg.get("max_iter", 40)),
    }
    optimizer = make_optimizer_from_config(calc, bands, lm_kwargs)

    print("IR / optical thin-film optimiser", flush=True)
    print(f"  input: {input_path}", flush=True)
    print(f"  angle: {angle_deg} deg  engine: {cfg.get('rt_engine', 'tmm')}", flush=True)
    print(f"  substrate_model: {substrate_model}", flush=True)
    print(f"  default materials: {', '.join(DEFAULT_MATERIALS)}", flush=True)
    print(f"  library materials: {', '.join(sorted(dsp.MATERIALS))}", flush=True)

    # Spectra for plotting use a denser grid over the full plot window.
    lo_nm, hi_nm = plot_range_nm(cfg, bands)
    plot_step = float(cfg.get("plot_step_nm", 5))
    plot_wls = [_nm(x) for x in dense_grid_nm(lo_nm, hi_nm, plot_step)]

    rt_kw = dict(
        incident=lm_kwargs["incident"],
        substrate=lm_kwargs["substrate"],
        substrate_thickness=lm_kwargs["substrate_thickness"],
        exit_medium=lm_kwargs["exit_medium"],
        polarization=lm_kwargs["polarization"],
        substrate_model=substrate_model,
    )
    R0, T0 = calc.spectrum(layers0, plot_wls, theta0, **rt_kw)
    _, _, ok0, report0 = optimizer.evaluate(layers0)
    print_report("Before optimisation", layers0, report0, ok0)

    use_needle = bool(opt_cfg.get("use_needle", True))
    if use_needle:
        synth = NeedleSynthesizer(
            optimizer,
            high_index=opt_cfg.get("high_index", "tio2"),
            low_index=opt_cfg.get("low_index", "sio2"),
            max_layers=int(opt_cfg.get("max_layers", 24)),
            max_add_rounds=int(opt_cfg.get("max_add_rounds", 20)),
        )
        result = synth.run(layers0, verbose=True)
        layers1 = result.layers
        ok1 = result.specs_ok
        report1 = result.report
    else:
        print("  LM thickness optimisation only (fixed layer count)")
        lm = optimizer.optimize(layers0, verbose=True)
        layers1 = lm.layers
        _, _, ok1, report1 = optimizer.evaluate(layers1)

    print_report("After optimisation", layers1, report1, ok1)

    R1, T1 = calc.spectrum(layers1, plot_wls, theta0, **rt_kw)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "spectrum_before_after.csv")
    stack_path = os.path.join(out_dir, "stack_optimised.json")
    plot_path = os.path.join(out_dir, "rt_and_n_before_after.png")

    write_spectrum_csv(csv_path, plot_wls, R0, T0, R1, T1)
    write_stack_json(
        stack_path,
        layers1,
        {
            "input": os.path.abspath(input_path),
            "incident_angle_deg": angle_deg,
            "specs_satisfied": ok1,
            "rt_engine": cfg.get("rt_engine", "tmm"),
        },
    )
    mats = []
    for m, _ in layers1:
        if m not in mats:
            mats.append(m)
    for m in ("glass", "air"):
        if m not in mats:
            mats.append(m)
    plot_results(plot_path, plot_wls, R0, T0, R1, T1, bands, mats)

    print(f"\n  wrote {csv_path}")
    print(f"  wrote {stack_path}")
    print(f"  wrote {plot_path}")
    return 0 if ok1 else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Optimise multilayer film thicknesses from an input JSON."
    )
    ap.add_argument(
        "input",
        nargs="?",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "examples",
            "example_vis_pass_ir_reflect.json",
        ),
        help="path to input JSON (default: example_vis_pass_ir_reflect.json)",
    )
    args = ap.parse_args(argv)
    cfg = load_input(args.input)
    return run(cfg, args.input)


if __name__ == "__main__":
    raise SystemExit(main())
