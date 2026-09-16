#!/usr/bin/env python3
"""Batch-run ``optimize_film.py`` over layer counts and H/L stacking modes.

Generates an alternating high/low-index initial stack for each job, writes a
per-job config with ``output_dir`` set, then invokes ``optimize_film.py``.

Modes:

  - ``hlh`` / ``高低高``: coating starts with high-index (H L H L …)
  - ``lhl`` / ``低高低``: coating starts with low-index  (L H L H …)

Example::

    python3 sim/batch_optimize.py \\
        --layers 5 7 9 \\
        --mode hlh \\
        --output-dir sim/out/batch_hlh

    python3 sim/batch_optimize.py \\
        --layers 8 \\
        --mode both \\
        -o sim/out/compare_modes \\
        --config sim/examples/example_optimize_film.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from copy import deepcopy
from typing import Sequence

# Default materials match example_stack.txt / needle.py conventions.
_DEFAULT_HIGH = {"name": "tio2", "n": 2.40, "k": 0.0, "thickness_nm": 55.0}
_DEFAULT_LOW = {"name": "sio2", "n": 1.46, "k": 0.0, "thickness_nm": 85.0}
_DEFAULT_INCIDENT = {"name": "air", "n": 1.00, "k": 0.0, "thickness_nm": 0.0}
_DEFAULT_SUBSTRATE = {"name": "glass", "n": 1.52, "k": 0.0, "thickness_nm": 0.0}

_MODE_ALIASES = {
    "hlh": "hlh",
    "h": "hlh",
    "高低高": "hlh",
    "high-low-high": "hlh",
    "high_low_high": "hlh",
    "lhl": "lhl",
    "l": "lhl",
    "低高低": "lhl",
    "low-high-low": "lhl",
    "low_high_low": "lhl",
    "both": "both",
    "all": "both",
}


def _normalize_mode(raw: str) -> str:
    key = str(raw).strip().lower()
    # Keep CJK aliases as-is for lookup before lowercasing breaks them.
    for alias, canon in _MODE_ALIASES.items():
        if raw.strip() == alias or key == alias.lower():
            return canon
    raise ValueError(
        f"unknown mode {raw!r}; use hlh/高低高, lhl/低高低, or both"
    )


def _parse_layers(values: Sequence[str]) -> list[int]:
    """Accept ``5 7 9``, ``5,7,9``, or ``5-9`` (inclusive range)."""
    out: list[int] = []
    for raw in values:
        for part in str(raw).replace(",", " ").split():
            part = part.strip()
            if not part:
                continue
            if "-" in part and part.count("-") == 1 and not part.startswith("-"):
                a, b = part.split("-", 1)
                lo, hi = int(a), int(b)
                if hi < lo:
                    lo, hi = hi, lo
                out.extend(range(lo, hi + 1))
            else:
                out.append(int(part))
    if not out:
        raise ValueError("need at least one layer count")
    for n in out:
        if n < 1:
            raise ValueError(f"layer count must be >= 1, got {n}")
    # Preserve order, drop duplicates.
    seen: set[int] = set()
    uniq: list[int] = []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def build_coating(
    n_layers: int,
    mode: str,
    *,
    high: dict,
    low: dict,
) -> list[dict]:
    """Return alternating coating rows (material dicts with thickness_nm)."""
    first_high = mode == "hlh"
    rows: list[dict] = []
    for i in range(n_layers):
        use_high = (i % 2 == 0) if first_high else (i % 2 == 1)
        src = high if use_high else low
        rows.append(dict(src))
    return rows


def write_stack_txt(
    path: str,
    *,
    n_layers: int,
    mode: str,
    high: dict,
    low: dict,
    incident: dict,
    substrate: dict,
) -> None:
    films = build_coating(n_layers, mode, high=high, low=low)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = [
        f"# auto-generated: n_layers={n_layers} mode={mode}",
        "# index  material  thickness_nm  n  k",
        f"0  {incident['name']}  {incident['thickness_nm']:g}  "
        f"{incident['n']:g}  {incident['k']:g}",
    ]
    for i, film in enumerate(films, start=1):
        lines.append(
            f"{i}  {film['name']}  {film['thickness_nm']:g}  "
            f"{film['n']:g}  {film['k']:g}"
        )
    last_idx = n_layers + 1
    lines.append(
        f"{last_idx}  {substrate['name']}  {substrate['thickness_nm']:g}  "
        f"{substrate['n']:g}  {substrate['k']:g}"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def load_base_config(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return cfg


def job_subdir_name(n_layers: int, mode: str) -> str:
    return f"n{n_layers}_{mode}"


def run_one(
    *,
    optimize_script: str,
    python_exe: str,
    stack_path: str,
    config_path: str,
    dry_run: bool,
) -> int:
    cmd = [python_exe, optimize_script, stack_path, config_path]
    print(f"\n=== {' '.join(cmd)} ===", flush=True)
    if dry_run:
        print("  (dry-run: skipped)", flush=True)
        return 0
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(here, "examples", "example_optimize_film.json")
    optimize_script = os.path.join(here, "optimize_film.py")

    ap = argparse.ArgumentParser(
        description="Batch-run optimize_film.py for given layer counts "
        "and high-low (hlh) / low-high (lhl) stacking modes."
    )
    ap.add_argument(
        "--layers",
        "-n",
        nargs="+",
        required=True,
        metavar="N",
        help="coating layer counts: e.g. 5 7 9, or 5,7,9, or 5-9",
    )
    ap.add_argument(
        "--mode",
        "-m",
        default="hlh",
        help="hlh/高低高, lhl/低高低, or both (default: hlh)",
    )
    ap.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="base output directory; each job writes under n{N}_{mode}/",
    )
    ap.add_argument(
        "--config",
        "-c",
        default=default_config,
        help=f"base optimize JSON (default: {default_config})",
    )
    ap.add_argument(
        "--work-dir",
        default=None,
        help="directory for generated stacks/configs "
        "(default: <output-dir>/_inputs)",
    )
    ap.add_argument(
        "--high-n",
        type=float,
        default=_DEFAULT_HIGH["n"],
        help=f"high-index n (default: {_DEFAULT_HIGH['n']})",
    )
    ap.add_argument(
        "--low-n",
        type=float,
        default=_DEFAULT_LOW["n"],
        help=f"low-index n (default: {_DEFAULT_LOW['n']})",
    )
    ap.add_argument(
        "--high-thickness-nm",
        type=float,
        default=_DEFAULT_HIGH["thickness_nm"],
        help=f"initial H thickness nm (default: {_DEFAULT_HIGH['thickness_nm']})",
    )
    ap.add_argument(
        "--low-thickness-nm",
        type=float,
        default=_DEFAULT_LOW["thickness_nm"],
        help=f"initial L thickness nm (default: {_DEFAULT_LOW['thickness_nm']})",
    )
    ap.add_argument(
        "--high-material",
        default=_DEFAULT_HIGH["name"],
        help=f"high-index material name (default: {_DEFAULT_HIGH['name']})",
    )
    ap.add_argument(
        "--low-material",
        default=_DEFAULT_LOW["name"],
        help=f"low-index material name (default: {_DEFAULT_LOW['name']})",
    )
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch optimize_film.py",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="write inputs only; do not run optimisation",
    )
    ap.add_argument(
        "--stop-on-error",
        action="store_true",
        help="abort the batch if one job fails (default: continue)",
    )
    args = ap.parse_args(argv)

    try:
        layer_counts = _parse_layers(args.layers)
        mode_sel = _normalize_mode(args.mode)
    except ValueError as exc:
        ap.error(str(exc))

    modes = ["hlh", "lhl"] if mode_sel == "both" else [mode_sel]

    out_base = os.path.abspath(
        args.output_dir
        if os.path.isabs(args.output_dir)
        else os.path.join(os.getcwd(), args.output_dir)
    )
    work_dir = os.path.abspath(
        args.work_dir
        if args.work_dir
        else os.path.join(out_base, "_inputs")
    )
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(out_base, exist_ok=True)

    high = {
        "name": args.high_material,
        "n": float(args.high_n),
        "k": 0.0,
        "thickness_nm": float(args.high_thickness_nm),
    }
    low = {
        "name": args.low_material,
        "n": float(args.low_n),
        "k": 0.0,
        "thickness_nm": float(args.low_thickness_nm),
    }

    base_cfg = load_base_config(os.path.abspath(args.config))

    print(f"batch optimize")
    print(f"  layers: {layer_counts}")
    print(f"  modes:  {modes}")
    print(f"  output: {out_base}")
    print(f"  inputs: {work_dir}")
    print(f"  config: {os.path.abspath(args.config)}")

    jobs: list[tuple[int, str, str, str]] = []
    for n_layers in layer_counts:
        for mode in modes:
            tag = job_subdir_name(n_layers, mode)
            job_out = os.path.join(out_base, tag)
            stack_path = os.path.join(work_dir, f"{tag}_stack.txt")
            config_path = os.path.join(work_dir, f"{tag}_config.json")

            write_stack_txt(
                stack_path,
                n_layers=n_layers,
                mode=mode,
                high=high,
                low=low,
                incident=_DEFAULT_INCIDENT,
                substrate=_DEFAULT_SUBSTRATE,
            )

            cfg = deepcopy(base_cfg)
            # Absolute path so resolve-relative-to-stack does not matter.
            cfg["output_dir"] = job_out
            cfg["comment"] = (
                f"batch job: n_layers={n_layers} mode={mode} "
                f"(from {os.path.basename(args.config)})"
            )
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2, ensure_ascii=False)
                fh.write("\n")

            jobs.append((n_layers, mode, stack_path, config_path))
            print(f"  prepared {tag} -> {job_out}")

    failures: list[str] = []
    for n_layers, mode, stack_path, config_path in jobs:
        tag = job_subdir_name(n_layers, mode)
        rc = run_one(
            optimize_script=optimize_script,
            python_exe=args.python,
            stack_path=stack_path,
            config_path=config_path,
            dry_run=args.dry_run,
        )
        if rc != 0:
            failures.append(tag)
            print(f"  FAILED {tag} (exit {rc})", flush=True)
            if args.stop_on_error:
                break

    print()
    if failures:
        print(f"done with {len(failures)} failure(s): {', '.join(failures)}")
        return 1
    print(f"done: {len(jobs)} job(s) ok under {out_base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
