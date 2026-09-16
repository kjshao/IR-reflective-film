#!/usr/bin/env python3
"""Generate an optical initial stack and run variable-layer optimization."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from multistart_optimize import prepare_inputs
from optimize_film import run


def _layer_search_mapping(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("layer_search", {})
    if raw is True:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("layer_search must be a JSON object or true")
    return dict(raw)


def _initial_layer_count(
    config: dict[str, Any],
    layer_search: dict[str, Any],
    override: int | None,
) -> int:
    stack_init = config.get("stack_init", {})
    if not isinstance(stack_init, dict):
        raise ValueError("stack_init must be a JSON object")
    explicit = (
        override
        if override is not None
        else stack_init.get(
            "initial_layers", layer_search.get("initial_layers")
        )
    )
    raw_min = layer_search.get("min_layers")
    raw_max = layer_search.get("max_layers")
    if explicit is not None:
        count = int(explicit)
    elif raw_min is not None and raw_max is not None:
        count = int(round((int(raw_min) + int(raw_max)) / 2.0))
    elif raw_min is not None:
        count = max(int(raw_min), 8)
    elif raw_max is not None:
        count = min(int(raw_max), 8)
    else:
        count = 8
    if count < 1:
        raise ValueError("initial layer count must be >= 1")
    return count


def prepare_variable_layer_inputs(
    *,
    config_path: str,
    initial_layers_override: int | None = None,
    output_override: str | None = None,
    mode_override: str | None = None,
    method_override: str | None = None,
) -> tuple[str, str]:
    """Generate an initial H/L stack and an effective layer-search config."""
    config_path = os.path.abspath(config_path)
    with open(config_path, encoding="utf-8") as handle:
        config: dict[str, Any] = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a JSON object")

    layer_search = _layer_search_mapping(config)
    initial_layers = _initial_layer_count(
        config, layer_search, initial_layers_override
    )
    min_layers = int(
        layer_search.get("min_layers", max(1, initial_layers - 4))
    )
    max_layers = int(
        layer_search.get("max_layers", max(min_layers, initial_layers + 6))
    )
    if not (1 <= min_layers <= initial_layers <= max_layers):
        raise ValueError(
            "initial layer count must satisfy "
            "1 <= layer_search.min_layers <= initial_layers "
            "<= layer_search.max_layers"
        )

    stack_path, effective_path = prepare_inputs(
        n_layers=initial_layers,
        config_path=config_path,
        output_override=output_override,
        mode_override=mode_override,
        method_override=method_override,
    )
    with open(effective_path, encoding="utf-8") as handle:
        effective: dict[str, Any] = json.load(handle)
    stack_init = effective.get("stack_init", {})
    high = str(stack_init.get("high_material", "tio2"))
    low = str(stack_init.get("low_material", "sio2"))
    layer_search["enabled"] = True
    layer_search["min_layers"] = min_layers
    layer_search["max_layers"] = max_layers
    layer_search.setdefault("materials", [high, low])
    layer_search["initial_layers"] = initial_layers
    effective["layer_search"] = layer_search
    effective["generated_variable_layer_stack"] = {
        "initial_layers": initial_layers,
        "min_layers": min_layers,
        "max_layers": max_layers,
        "mode": effective["generated_stack"]["mode"],
        "design_wavelength_nm": effective["generated_stack"][
            "design_wavelength_nm"
        ],
    }
    with open(effective_path, "w", encoding="utf-8") as handle:
        json.dump(effective, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return stack_path, effective_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a quarter-wave H/L seed and run Pareto-beam "
            "variable-layer optimization."
        )
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="variable-layer JSON configuration",
    )
    parser.add_argument(
        "--initial-layers",
        "-n",
        type=int,
        default=None,
        help=(
            "override initial coating layer count; otherwise use "
            "stack_init.initial_layers or the midpoint of the search range"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("hlh", "lhl"),
        default=None,
        help="override initial alternating material order",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="override configuration output_dir",
    )
    parser.add_argument(
        "--method",
        choices=("multistart", "auto"),
        default=None,
        help="override the full-fidelity refinement method",
    )
    args = parser.parse_args(argv)

    stack_path, effective_config = prepare_variable_layer_inputs(
        config_path=args.config,
        initial_layers_override=args.initial_layers,
        output_override=args.output_dir,
        mode_override=args.mode,
        method_override=args.method,
    )
    print("Generated variable-layer optimization inputs")
    print(f"  stack:  {stack_path}")
    print(f"  config: {effective_config}")
    return run(stack_path, effective_config)


if __name__ == "__main__":
    raise SystemExit(main())
