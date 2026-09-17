"""Compute R/T from a plain-text stack file; wavelength range via CLI.

Usage::

    python3 sim/plot_rt_txt.py sim/examples/example_stack.txt 400 1800
    python3 sim/plot_rt_txt.py stack.txt 420 1800 --step 5 --angle 0
    python3 sim/plot_rt_txt.py stack.txt 300 1800 \
        --window 400 700 --window 780 1800

Stack file format (whitespace-separated; ``#`` comments allowed)::

    index  material  thickness_nm  n  k

  - first line: incident medium (thickness ignored)
  - middle lines: coherent coating layers (thickness in nm)
  - last line: substrate / exit medium (thickness ignored; semi-infinite)

``n`` / ``k`` are wavelength-independent when ``nk_source=fixed``.
With ``nk_source=library`` (default for ``optimize_film.py``), those columns
are ignored and optical constants come from ``dispersion.py`` by material name.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dispersion as dsp
import tmm
from lm_optimizer import BandSpec
from plot_rt import close_all_figures, configure_matplotlib, dense_grid_nm, plot_rt, write_spectrum_csv
from rt_calculator import make_calculator

_NM = 1e-9


@dataclass
class StackRow:
    index: int
    material: str
    thickness_nm: float
    n: float
    k: float

    @property
    def N(self) -> complex:
        return complex(self.n, self.k)

    @property
    def thickness_m(self) -> float:
        return self.thickness_nm * _NM


def _parse_line(line: str, lineno: int) -> StackRow | None:
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    # Allow comma or whitespace separators.
    parts = raw.replace(",", " ").split()
    if len(parts) < 5:
        raise ValueError(
            f"line {lineno}: need index material thickness_nm n k, got {raw!r}"
        )
    return StackRow(
        index=int(float(parts[0])),
        material=parts[1],
        thickness_nm=float(parts[2]),
        n=float(parts[3]),
        k=float(parts[4]),
    )


def load_stack_txt(path: str) -> tuple[StackRow, list[StackRow], StackRow]:
    """Return (incident, coating_layers, substrate)."""
    rows: list[StackRow] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            row = _parse_line(line, i)
            if row is not None:
                rows.append(row)
    if len(rows) < 3:
        raise ValueError(
            f"{path}: need at least 3 rows "
            f"(incident + ≥1 film + substrate), got {len(rows)}"
        )
    return rows[0], rows[1:-1], rows[-1]


def compute_spectrum(
    incident: StackRow,
    films: list[StackRow],
    substrate: StackRow,
    wavelengths_m: list[float],
    theta0: float = 0.0,
    polarization: str = "unpolarized",
    *,
    nk_source: str = "fixed",
) -> tuple[list[float], list[float]]:
    """Semi-infinite substrate: coherent coating between incident and substrate.

    ``nk_source='fixed'`` uses stack-file n,k. ``'library'`` uses the dispersion
    database by material name (wavelength-dependent).
    """
    src = str(nk_source).strip().lower()
    if src in ("library", "lib", "dispersion", "database", "db"):
        for row in (incident, *films, substrate):
            key = dsp.normalize_material_name(row.material)
            if key not in dsp.MATERIALS:
                raise KeyError(
                    f"material {row.material!r} not in dispersion library; "
                    f"known: {sorted(dsp.MATERIALS)}"
                )
        calc = make_calculator()
        return calc.spectrum(
            [(f.material, f.thickness_m) for f in films],
            wavelengths_m,
            theta0,
            incident=incident.material,
            substrate=substrate.material,
            polarization=polarization,
            substrate_model="semi_infinite",
        )
    if src not in ("fixed", "stack", "constant", "input", "file"):
        raise ValueError(f"nk_source must be library|fixed, got {nk_source!r}")

    coating = [(f.N, f.thickness_m) for f in films]
    rs, ts = [], []
    for wl in wavelengths_m:
        stack = [(incident.N, 0.0), *coating, (substrate.N, 0.0)]
        if polarization in ("unpolarized", "avg", "average"):
            r, t = tmm.unpolarised(tmm.coherent_rt, stack, wl, theta0)
        elif polarization in ("s", "p"):
            r, t = tmm.coherent_rt(stack, wl, theta0, polarization)
        else:
            raise ValueError(f"polarization must be s/p/unpolarized, got {polarization!r}")
        rs.append(r)
        ts.append(t)
    return rs, ts


def validate_windows(
    windows_nm: list[tuple[float, float]] | None,
    wl_lo_nm: float,
    wl_hi_nm: float,
) -> list[tuple[float, float]]:
    """Validate optional closed wavelength windows inside the plotted range."""
    result: list[tuple[float, float]] = []
    for lo, hi in windows_nm or []:
        lo, hi = float(lo), float(hi)
        if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
            raise SystemExit(f"window invalid: {lo:g} .. {hi:g}")
        if lo < wl_lo_nm or hi > wl_hi_nm:
            raise SystemExit(
                f"window {lo:g}–{hi:g} nm is outside plotted range "
                f"{wl_lo_nm:g}–{wl_hi_nm:g} nm"
            )
        result.append((lo, hi))
    return result


def window_average_stats(
    wavelengths_m: list[float],
    R: list[float],
    T: list[float],
    windows_nm: list[tuple[float, float]],
) -> list[dict[str, float | int]]:
    """Mean sampled R and T in each closed wavelength window."""
    rows: list[dict[str, float | int]] = []
    for lo_nm, hi_nm in windows_nm:
        indices = [
            i
            for i, wavelength in enumerate(wavelengths_m)
            if lo_nm <= wavelength / _NM <= hi_nm
        ]
        if not indices:
            raise SystemExit(
                f"window {lo_nm:g}–{hi_nm:g} nm contains no sampled wavelengths"
            )
        rows.append(
            {
                "wl_lo_nm": lo_nm,
                "wl_hi_nm": hi_nm,
                "sample_count": len(indices),
                "R_mean": sum(R[i] for i in indices) / len(indices),
                "T_mean": sum(T[i] for i in indices) / len(indices),
            }
        )
    return rows


def write_window_stats_csv(
    path: str,
    rows: list[dict[str, float | int]],
) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("window,wl_lo_nm,wl_hi_nm,sample_count,R_mean,T_mean\n")
        for index, row in enumerate(rows, 1):
            fh.write(
                f"{index},{row['wl_lo_nm']:.6g},{row['wl_hi_nm']:.6g},"
                f"{row['sample_count']},{row['R_mean']:.9g},{row['T_mean']:.9g}\n"
            )


def run(
    stack_path: str,
    wl_lo_nm: float,
    wl_hi_nm: float,
    *,
    step_nm: float = 5.0,
    angle_deg: float = 0.0,
    polarization: str = "unpolarized",
    out_dir: str | None = None,
    nk_source: str = "library",
    windows_nm: list[tuple[float, float]] | None = None,
) -> int:
    if wl_hi_nm <= wl_lo_nm:
        raise SystemExit(f"wavelength range invalid: {wl_lo_nm} .. {wl_hi_nm}")
    if step_nm <= 0:
        raise SystemExit(f"step must be > 0, got {step_nm}")
    windows = validate_windows(windows_nm, wl_lo_nm, wl_hi_nm)

    incident, films, substrate = load_stack_txt(stack_path)
    plot_wls = [x * _NM for x in dense_grid_nm(wl_lo_nm, wl_hi_nm, step_nm)]
    theta0 = math.radians(angle_deg)
    R, T = compute_spectrum(
        incident,
        films,
        substrate,
        plot_wls,
        theta0,
        polarization,
        nk_source=nk_source,
    )

    if out_dir is None:
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "out", "plot_rt_txt"
        )
    elif not os.path.isabs(out_dir):
        out_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(stack_path)), out_dir)
        )

    total_nm = sum(f.thickness_nm for f in films)
    src = str(nk_source).strip().lower()
    use_lib = src in ("library", "lib", "dispersion", "database", "db")
    print("Film R/T spectrum (text stack)")
    print(f"  stack: {stack_path}")
    print(f"  nk_source: {'library' if use_lib else 'fixed'}")
    print(f"  wavelength: {wl_lo_nm:g}–{wl_hi_nm:g} nm  step={step_nm:g} nm")
    print(f"  angle: {angle_deg:g} deg  pol: {polarization}")
    if use_lib:
        print(f"  incident: {incident.material}  (n,k from dispersion library)")
    else:
        print(f"  incident: {incident.material}  n={incident.n:g}  k={incident.k:g}")
    print(f"  {len(films)} film layers, total thickness {total_nm:.2f} nm")
    for f in films:
        if use_lib:
            print(f"    {f.index:2d}. {f.material:<10} {f.thickness_nm:8.2f} nm")
        else:
            print(
                f"    {f.index:2d}. {f.material:<10} "
                f"{f.thickness_nm:8.2f} nm  n={f.n:g}  k={f.k:g}"
            )
    if use_lib:
        print(f"  substrate: {substrate.material}  (n,k from dispersion library)")
    else:
        print(f"  substrate: {substrate.material}  n={substrate.n:g}  k={substrate.k:g}")

    window_rows = window_average_stats(plot_wls, R, T, windows)
    if window_rows:
        print("  window averages:")
        for row in window_rows:
            print(
                f"    {row['wl_lo_nm']:g}–{row['wl_hi_nm']:g} nm: "
                f"R_mean={100 * row['R_mean']:.3f}%  "
                f"T_mean={100 * row['T_mean']:.3f}%  "
                f"(samples={row['sample_count']})"
            )

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "spectrum.csv")
    plot_path = os.path.join(out_dir, "rt_spectrum.png")
    window_csv_path = os.path.join(out_dir, "window_stats.csv")
    write_spectrum_csv(csv_path, plot_wls, R, T)
    if window_rows:
        write_window_stats_csv(window_csv_path, window_rows)
    plot_rt(
        plot_path,
        plot_wls,
        R,
        T,
        bands=[
            BandSpec(wl_lo=lo_nm * _NM, wl_hi=hi_nm * _NM)
            for lo_nm, hi_nm in windows
        ],
        title=f"{os.path.basename(stack_path)}  ({wl_lo_nm:g}–{wl_hi_nm:g} nm)",
        layers=[(f.material, f.thickness_m) for f in films],
    )
    print(f"\n  wrote {csv_path}")
    if window_rows:
        print(f"  wrote {window_csv_path}")
    print(f"  wrote {plot_path}")
    close_all_figures()
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_matplotlib()
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Read a plain-text film stack (n, k per layer), "
        "compute R/T over a CLI wavelength range, and plot with matplotlib."
    )
    ap.add_argument(
        "stack",
        nargs="?",
        default=os.path.join(here, "examples", "example_stack.txt"),
        help="text stack file (default: examples/example_stack.txt)",
    )
    ap.add_argument(
        "wl_min",
        nargs="?",
        type=float,
        default=400.0,
        help="plot wavelength start (nm)",
    )
    ap.add_argument(
        "wl_max",
        nargs="?",
        type=float,
        default=1800.0,
        help="plot wavelength end (nm)",
    )
    ap.add_argument("--step", type=float, default=5.0, help="wavelength step (nm)")
    ap.add_argument("--angle", type=float, default=0.0, help="incidence angle (deg)")
    ap.add_argument(
        "--pol",
        default="unpolarized",
        choices=("unpolarized", "s", "p"),
        help="polarization",
    )
    ap.add_argument(
        "-o",
        "--out-dir",
        default=None,
        help="output directory (default: sim/out/plot_rt_txt)",
    )
    ap.add_argument(
        "--nk-source",
        default="library",
        choices=("library", "fixed"),
        help="library=dispersion DB by material name (default); "
        "fixed=use n,k columns from the stack file",
    )
    ap.add_argument(
        "--window",
        nargs=2,
        action="append",
        type=float,
        default=None,
        metavar=("WL_MIN", "WL_MAX"),
        help="closed wavelength window in nm for mean R/T; may be repeated",
    )
    args = ap.parse_args(argv)
    return run(
        args.stack,
        args.wl_min,
        args.wl_max,
        step_nm=args.step,
        angle_deg=args.angle,
        polarization=args.pol,
        out_dir=args.out_dir,
        nk_source=args.nk_source,
        windows_nm=args.window,
    )


if __name__ == "__main__":
    raise SystemExit(main())
