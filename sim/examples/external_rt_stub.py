#!/usr/bin/env python3
"""Minimal external R/T engine stub for rt_engine=external.

Reads the stack JSON written by ExternalRTCalculator and writes wavelength,R,T CSV.
Uses the in-repo TMM solver — replace the body with a call to another program.

    python3 sim/examples/external_rt_stub.py --input stack.json --output rt.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rt_calculator import TMMCalculator


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    with open(args.input, encoding="utf-8") as fh:
        payload = json.load(fh)
    layers = [(L["material"], float(L["thickness_m"])) for L in payload["layers"]]
    calc = TMMCalculator()
    R, T = calc.spectrum(
        layers,
        payload["wavelengths_m"],
        float(payload.get("theta0_rad", 0.0)),
        incident=payload.get("incident", "air"),
        substrate=payload.get("substrate", "glass"),
        substrate_thickness=float(payload.get("substrate_thickness_m", 0.7e-3)),
        exit_medium=payload.get("exit_medium", "air"),
        polarization=payload.get("polarization", "unpolarized"),
        substrate_model=payload.get("substrate_model", "semi_infinite"),
    )
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write("wavelength_m,R,T\n")
        for wl, r, t in zip(payload["wavelengths_m"], R, T):
            fh.write(f"{wl:.12e},{r:.8f},{t:.8f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
