"""Compute reflectance / transmittance from a JSON stack and plot the spectra.

Usage::

    python3 sim/plot_rt.py sim/examples/example_plot_rt.json
    python3 sim/plot_rt.py sim/out/optimize_example/stack_optimised.json \\
        --bands-from sim/examples/example_vis_pass_ir_reflect.json

Input JSON fields:
  - layers: [{material, thickness_nm}, ...]  (required unless ``seed`` is set)
  - bands:  [{wavelength_nm: [lo, hi], ...}, ...]  (optional; shaded on plot)
  - plot_wavelength_nm, plot_step_nm, incident_angle_deg, substrate, ...
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
from lm_optimizer import BandSpec
from rt_calculator import make_calculator, material_index

_NM = 1e-9

BAND_BG_COLORS = (
    "#cfe8ff",
    "#ffe4cc",
    "#d4f0d4",
    "#f0e0ff",
    "#fff3c4",
    "#ffd6e0",
    "#d0f5f0",
    "#e8e4d8",
)


def band_bg_color(index: int) -> str:
    return BAND_BG_COLORS[index % len(BAND_BG_COLORS)]


def shade_bands(ax, bands: list[BandSpec]) -> None:
    for i, b in enumerate(bands):
        ax.axvspan(
            b.wl_lo / _NM,
            b.wl_hi / _NM,
            color=band_bg_color(i),
            alpha=0.45,
            lw=0,
            zorder=0,
        )
    edges_nm = sorted(
        {b.wl_lo / _NM for b in bands} | {b.wl_hi / _NM for b in bands}
    )
    for x in edges_nm:
        ax.axvline(
            x,
            color="0.35",
            linestyle="--",
            lw=0.9,
            alpha=0.75,
            zorder=1,
        )


def load_input(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def parse_bands(raw: list[dict]) -> list[BandSpec]:
    bands = []
    for b in raw:
        w = b["wavelength_nm"]
        if len(w) != 2:
            raise ValueError("each band needs wavelength_nm: [lo, hi]")
        bands.append(
            BandSpec(
                wl_lo=float(w[0]) * _NM,
                wl_hi=float(w[1]) * _NM,
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
            d = float(layer["thickness_nm"]) * _NM
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
    centres = [float(c) * _NM for c in seed.get("centres_nm", [900, 1200, 1500])]
    raw_p = seed.get("periods_per_centre", 3)
    if isinstance(raw_p, list):
        if len(raw_p) != len(centres):
            raise ValueError("periods_per_centre list must match centres_nm")
        periods_list = [int(p) for p in raw_p]
    else:
        periods_list = [int(raw_p)] * len(centres)
    cell = [str(m).lower() for m in seed.get("cell", ["tio2", "sio2"])]
    layers: list[tuple[str, float]] = []
    for lam0, periods in zip(centres, periods_list):
        for _ in range(periods):
            for mat in cell:
                if mat not in dsp.MATERIALS:
                    raise KeyError(f"seed material '{mat}' not in library")
                n = dsp.MATERIALS[mat](lam0).real
                layers.append((mat, 0.25 * lam0 / max(n, 1.01)))
    return layers


def plot_range_nm(cfg: dict, bands: list[BandSpec]) -> tuple[float, float]:
    if "plot_wavelength_nm" in cfg:
        lo, hi = cfg["plot_wavelength_nm"]
        return float(lo), float(hi)
    lo = min(b.wl_lo for b in bands) / _NM
    hi = max(b.wl_hi for b in bands) / _NM
    return lo, hi


def dense_grid_nm(lo_nm: float, hi_nm: float, step_nm: float) -> list[float]:
    n = max(1, int(round((hi_nm - lo_nm) / step_nm)))
    return [lo_nm + i * (hi_nm - lo_nm) / n for i in range(n + 1)]


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

    wl_nm = [w / _NM for w in wavelengths_m]
    materials_to_show = materials_to_show or ["sio2", "tio2", "glass"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    ax = axes[0]
    ax.plot(wl_nm, [100 * r for r in R_before], "--", color="C0", label="R before")
    ax.plot(wl_nm, [100 * r for r in R_after], "-", color="C0", label="R after")
    ax.plot(wl_nm, [100 * t for t in T_before], "--", color="C1", label="T before")
    ax.plot(wl_nm, [100 * t for t in T_after], "-", color="C1", label="T after")
    shade_bands(ax, bands)
    for b in bands:
        if b.R_min is not None:
            ax.hlines(
                100 * b.R_min,
                b.wl_lo / _NM,
                b.wl_hi / _NM,
                colors="C0",
                linestyles=":",
                lw=1,
            )
        if b.R_max is not None:
            ax.hlines(
                100 * b.R_max,
                b.wl_lo / _NM,
                b.wl_hi / _NM,
                colors="C0",
                linestyles=":",
                lw=1,
            )
        if b.T_min is not None:
            ax.hlines(
                100 * b.T_min,
                b.wl_lo / _NM,
                b.wl_hi / _NM,
                colors="C1",
                linestyles=":",
                lw=1,
            )
        if b.T_max is not None:
            ax.hlines(
                100 * b.T_max,
                b.wl_lo / _NM,
                b.wl_hi / _NM,
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
    shade_bands(ax, bands)
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


def band_stats(
    wavelengths_m: list[float],
    R: list[float],
    T: list[float],
    bands: list[BandSpec],
) -> list[dict[str, Any]]:
    """Mean / min / max of R and T inside each band."""
    rows = []
    for b in bands:
        idx = [
            i
            for i, wl in enumerate(wavelengths_m)
            if b.wl_lo <= wl <= b.wl_hi
        ]
        if not idx:
            continue
        rr = [R[i] for i in idx]
        tt = [T[i] for i in idx]
        rows.append(
            {
                "wl_nm": (b.wl_lo / _NM, b.wl_hi / _NM),
                "R_mean": sum(rr) / len(rr),
                "R_min": min(rr),
                "R_max": max(rr),
                "T_mean": sum(tt) / len(tt),
                "T_min": min(tt),
                "T_max": max(tt),
            }
        )
    return rows


def write_band_stats_csv(
    path: str,
    wavelengths_m: list[float],
    R: list[float],
    T: list[float],
    bands: list[BandSpec],
) -> None:
    """Write per-band R/T mean/min/max table as CSV."""
    rows = band_stats(wavelengths_m, R, T, bands)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            "band,wl_lo_nm,wl_hi_nm,"
            "R_mean,R_min,R_max,T_mean,T_min,T_max\n"
        )
        for i, row in enumerate(rows, 1):
            lo, hi = row["wl_nm"]
            fh.write(
                f"{i},{lo:.2f},{hi:.2f},"
                f"{row['R_mean']:.6f},{row['R_min']:.6f},{row['R_max']:.6f},"
                f"{row['T_mean']:.6f},{row['T_min']:.6f},{row['T_max']:.6f}\n"
            )


def write_spectrum_csv(
    path: str,
    wavelengths_m: list[float],
    R: list[float],
    T: list[float],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("wavelength_nm,R,T,A\n")
        for wl, r, t in zip(wavelengths_m, R, T):
            a = max(0.0, 1.0 - r - t)
            fh.write(f"{wl / _NM:.2f},{r:.6f},{t:.6f},{a:.6f}\n")


def plot_rt(
    path: str,
    wavelengths_m: list[float],
    R: list[float],
    T: list[float],
    bands: list[BandSpec],
    title: str = "Reflectance & transmittance",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting; pip install matplotlib"
        ) from exc

    wl_nm = [w / _NM for w in wavelengths_m]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax = axes[0]
    ax.plot(wl_nm, [100 * r for r in R], color="C0", label="R")
    ax.plot(wl_nm, [100 * t for t in T], color="C1", label="T")
    ax.plot(
        wl_nm,
        [100 * max(0.0, 1.0 - r - t) for r, t in zip(R, T)],
        color="C2",
        label="A ≈ 1−R−T",
        alpha=0.8,
    )
    shade_bands(ax, bands)
    for b in bands:
        for val, color in (
            (b.R_min, "C0"),
            (b.R_max, "C0"),
            (b.T_min, "C1"),
            (b.T_max, "C1"),
        ):
            if val is not None:
                ax.hlines(
                    100 * val,
                    b.wl_lo / _NM,
                    b.wl_hi / _NM,
                    colors=color,
                    linestyles=":",
                    lw=1,
                )
    ax.set_ylabel("R, T, A (%)")
    ax.set_ylim(-2, 105)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(wl_nm, [100 * r for r in R], color="C0", label="R")
    ax.plot(wl_nm, [100 * t for t in T], color="C1", label="T")
    shade_bands(ax, bands)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("R, T (%)")
    ax.set_ylim(-2, 105)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def resolve_layers(cfg: dict) -> list[tuple[str, float]]:
    if cfg.get("layers"):
        return parse_layers(cfg["layers"])
    if cfg.get("seed"):
        return build_chirped_seed(cfg["seed"])
    raise ValueError("input needs 'layers' (list) or 'seed'")


def run(cfg: dict, input_path: str, bands_cfg: dict | None = None) -> int:
    src = bands_cfg if bands_cfg is not None else cfg
    bands = parse_bands(src["bands"]) if src.get("bands") else []
    layers = resolve_layers(cfg)

    angle_deg = float(cfg.get("incident_angle_deg", 0.0))
    theta0 = math.radians(angle_deg)
    substrate_model = cfg.get("substrate_model", "semi_infinite")

    out_dir = cfg.get(
        "output_dir",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "plot_rt"),
    )
    if not os.path.isabs(out_dir):
        out_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(input_path)), out_dir)
        )

    if bands:
        lo_nm, hi_nm = plot_range_nm(cfg if "plot_wavelength_nm" in cfg else src, bands)
    else:
        lo_nm, hi_nm = cfg.get("plot_wavelength_nm", [400, 1800])
        lo_nm, hi_nm = float(lo_nm), float(hi_nm)
    plot_step = float(cfg.get("plot_step_nm", src.get("plot_step_nm", 5)))
    plot_wls = [x * _NM for x in dense_grid_nm(lo_nm, hi_nm, plot_step)]

    calc = make_calculator(
        cfg.get("rt_engine", "tmm"),
        cfg.get("external_command"),
        use_cuda=bool(cfg.get("use_cuda", False)),
    )
    rt_kw = dict(
        incident=cfg.get("incident_medium", "air"),
        substrate=cfg.get("substrate", "glass"),
        substrate_thickness=float(cfg.get("substrate_thickness_m", 0.7e-3)),
        exit_medium=cfg.get("exit_medium", "air"),
        polarization=cfg.get("polarization", "unpolarized"),
        substrate_model=substrate_model,
    )
    R, T = calc.spectrum(layers, plot_wls, theta0, **rt_kw)

    total_nm = sum(d for _, d in layers) / _NM
    print("Film R/T spectrum")
    print(f"  input: {input_path}")
    print(f"  angle: {angle_deg} deg  engine: {cfg.get('rt_engine', 'tmm')}")
    print(f"  substrate_model: {substrate_model}")
    print(f"  {len(layers)} layers, total thin-film thickness {total_nm:.1f} nm")
    for i, (m, d) in enumerate(layers, 1):
        print(f"    {i:2d}. {m:<8} {d / _NM:8.2f} nm")

    for row in band_stats(plot_wls, R, T, bands):
        lo, hi = row["wl_nm"]
        print(
            f"  {lo:.0f}-{hi:.0f} nm: "
            f"R mean/min/max = {100*row['R_mean']:.1f}/"
            f"{100*row['R_min']:.1f}/{100*row['R_max']:.1f}%  "
            f"T mean/min/max = {100*row['T_mean']:.1f}/"
            f"{100*row['T_min']:.1f}/{100*row['T_max']:.1f}%"
        )

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "spectrum.csv")
    plot_path = os.path.join(out_dir, "rt_spectrum.png")
    write_spectrum_csv(csv_path, plot_wls, R, T)
    plot_rt(plot_path, plot_wls, R, T, bands, title=os.path.basename(input_path))
    print(f"\n  wrote {csv_path}")
    print(f"  wrote {plot_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Read a layer stack (+ optional bands) from JSON, "
        "compute R/T with TMM, and plot with matplotlib."
    )
    ap.add_argument(
        "input",
        nargs="?",
        default=os.path.join(here, "examples", "example_plot_rt.json"),
        help="JSON with layers (or seed) and optional bands",
    )
    ap.add_argument(
        "--bands-from",
        default=None,
        help="optional second JSON that supplies the bands list "
        "(useful when plotting stack_optimised.json)",
    )
    args = ap.parse_args(argv)
    cfg = load_input(args.input)
    bands_cfg = load_input(args.bands_from) if args.bands_from else None
    return run(cfg, args.input, bands_cfg)


if __name__ == "__main__":
    raise SystemExit(main())
