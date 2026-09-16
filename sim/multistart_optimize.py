#!/usr/bin/env python3
"""Generate an alternating stack and optimize it with multi-start or auto."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import dispersion as dsp
from batch_optimize import write_stack_txt
from optimize_film import run


def _material_row(name: str, wavelength_m: float, thickness_nm: float) -> dict:
    index = dsp.material_n(name, wavelength_m)
    return {
        "name": name,
        "n": float(index.real),
        "k": float(index.imag),
        "thickness_nm": float(thickness_nm),
    }


def _quarter_wave_nm(material: str, wavelength_m: float) -> float:
    index = dsp.material_n(material, wavelength_m)
    if index.real <= 0.0:
        raise ValueError(f"{material}: refractive index must be positive")
    return wavelength_m * 1e9 / (4.0 * index.real)


def prepare_inputs(
    *,
    n_layers: int,
    config_path: str,
    output_override: str | None = None,
    mode_override: str | None = None,
    method_override: str | None = None,
) -> tuple[str, str]:
    """Write the generated stack and effective optimization configuration."""
    config_path = os.path.abspath(config_path)
    with open(config_path, encoding="utf-8") as fh:
        config: dict[str, Any] = json.load(fh)
    if n_layers < 1:
        raise ValueError("layers must be >= 1")
    method = str(method_override or config.get("method", "multistart")).lower()
    if method not in ("multistart", "auto"):
        raise ValueError(
            "generated-stack runner supports method 'multistart' or 'auto'"
        )

    stack_cfg = config.get("stack_init", {})
    if not isinstance(stack_cfg, dict):
        raise ValueError("stack_init must be a JSON object")
    mode = str(mode_override or stack_cfg.get("mode", "hlh")).lower()
    if mode not in ("hlh", "lhl"):
        raise ValueError("stack mode must be 'hlh' or 'lhl'")

    high_name = str(stack_cfg.get("high_material", "tio2"))
    low_name = str(stack_cfg.get("low_material", "sio2"))
    incident_name = str(stack_cfg.get("incident_medium", "air"))
    substrate_name = str(stack_cfg.get("substrate", "glass"))
    design_nm = float(stack_cfg.get("design_wavelength_nm", 1000.0))
    if design_nm <= 0.0:
        raise ValueError("stack_init.design_wavelength_nm must be positive")
    design_wavelength = design_nm * 1e-9

    high_nm = float(
        stack_cfg.get(
            "high_thickness_nm",
            _quarter_wave_nm(high_name, design_wavelength),
        )
    )
    low_nm = float(
        stack_cfg.get(
            "low_thickness_nm",
            _quarter_wave_nm(low_name, design_wavelength),
        )
    )
    high = _material_row(high_name, design_wavelength, high_nm)
    low = _material_row(low_name, design_wavelength, low_nm)
    incident = _material_row(incident_name, design_wavelength, 0.0)
    substrate = _material_row(substrate_name, design_wavelength, 0.0)

    raw_output = output_override or config.get(
        "output_dir", "../out/multistart_optimize"
    )
    output_dir = (
        os.path.abspath(raw_output)
        if os.path.isabs(raw_output)
        else os.path.abspath(os.path.join(os.path.dirname(config_path), raw_output))
    )
    input_dir = os.path.join(output_dir, "_inputs")
    os.makedirs(input_dir, exist_ok=True)

    stack_path = os.path.join(input_dir, f"stack_n{n_layers}_{mode}.txt")
    effective_config_path = os.path.join(
        input_dir, f"config_n{n_layers}_{mode}.json"
    )
    write_stack_txt(
        stack_path,
        n_layers=n_layers,
        mode=mode,
        high=high,
        low=low,
        incident=incident,
        substrate=substrate,
    )

    config["method"] = method
    config["output_dir"] = output_dir
    config["generated_stack"] = {
        "layers": n_layers,
        "mode": mode,
        "design_wavelength_nm": design_nm,
        "high_thickness_nm": high_nm,
        "low_thickness_nm": low_nm,
    }
    with open(effective_config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return stack_path, effective_config_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate an H/L stack and run multi-start or auto optimization."
    )
    parser.add_argument(
        "--layers", "-n", type=int, required=True, help="number of coating layers"
    )
    parser.add_argument(
        "--config", "-c", required=True, help="multi-start JSON configuration"
    )
    parser.add_argument(
        "--mode", choices=("hlh", "lhl"), default=None, help="override stack mode"
    )
    parser.add_argument(
        "--output-dir", "-o", default=None, help="override configuration output_dir"
    )
    parser.add_argument(
        "--method",
        choices=("multistart", "auto"),
        default=None,
        help="override configuration method",
    )
    args = parser.parse_args(argv)

    stack_path, effective_config = prepare_inputs(
        n_layers=args.layers,
        config_path=args.config,
        output_override=args.output_dir,
        mode_override=args.mode,
        method_override=args.method,
    )
    print("Generated-stack optimization inputs")
    print(f"  stack:  {stack_path}")
    print(f"  config: {effective_config}")
    return run(stack_path, effective_config)


if __name__ == "__main__":
    raise SystemExit(main())
