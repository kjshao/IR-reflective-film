"""Compute R/T from a plain-text stack file; wavelength range via CLI.

Usage::

    python3 sim/plot_rt_txt.py sim/examples/example_stack.txt 400 1800
    python3 sim/plot_rt_txt.py stack.txt 420 1800 --step 5 --angle 0

Stack file format (whitespace-separated; ``#`` comments allowed)::

    index  material  thickness_nm  n  k

  - first line: incident medium (thickness ignored)
  - middle lines: coherent coating layers (thickness in nm)
  - last line: substrate / exit medium (thickness ignored; semi-infinite)

``n`` is the refractive index and ``k`` is the extinction coefficient
(N = n + i·k). Values are taken as wavelength-independent.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tmm
from optimize_film import dense_grid_nm
from plot_rt import plot_rt, write_spectrum_csv

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
) -> tuple[list[float], list[float]]:
    """Semi-infinite substrate: coherent coating between incident and substrate."""
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


def run(
    stack_path: str,
    wl_lo_nm: float,
    wl_hi_nm: float,
    *,
    step_nm: float = 5.0,
    angle_deg: float = 0.0,
    polarization: str = "unpolarized",
    out_dir: str | None = None,
) -> int:
    if wl_hi_nm <= wl_lo_nm:
        raise SystemExit(f"wavelength range invalid: {wl_lo_nm} .. {wl_hi_nm}")
    if step_nm <= 0:
        raise SystemExit(f"step must be > 0, got {step_nm}")

    incident, films, substrate = load_stack_txt(stack_path)
    plot_wls = [x * _NM for x in dense_grid_nm(wl_lo_nm, wl_hi_nm, step_nm)]
    theta0 = math.radians(angle_deg)
    R, T = compute_spectrum(
        incident, films, substrate, plot_wls, theta0, polarization
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
    print("Film R/T spectrum (text stack)")
    print(f"  stack: {stack_path}")
    print(f"  wavelength: {wl_lo_nm:g}–{wl_hi_nm:g} nm  step={step_nm:g} nm")
    print(f"  angle: {angle_deg:g} deg  pol: {polarization}")
    print(f"  incident: {incident.material}  n={incident.n:g}  k={incident.k:g}")
    print(f"  {len(films)} film layers, total thickness {total_nm:.2f} nm")
    for f in films:
        print(
            f"    {f.index:2d}. {f.material:<10} "
            f"{f.thickness_nm:8.2f} nm  n={f.n:g}  k={f.k:g}"
        )
    print(f"  substrate: {substrate.material}  n={substrate.n:g}  k={substrate.k:g}")

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "spectrum.csv")
    plot_path = os.path.join(out_dir, "rt_spectrum.png")
    write_spectrum_csv(csv_path, plot_wls, R, T)
    plot_rt(
        plot_path,
        plot_wls,
        R,
        T,
        bands=[],
        title=f"{os.path.basename(stack_path)}  ({wl_lo_nm:g}–{wl_hi_nm:g} nm)",
    )
    print(f"\n  wrote {csv_path}")
    print(f"  wrote {plot_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
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
    args = ap.parse_args(argv)
    return run(
        args.stack,
        args.wl_min,
        args.wl_max,
        step_nm=args.step,
        angle_deg=args.angle,
        polarization=args.pol,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
