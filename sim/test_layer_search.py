"""Regression tests for Pareto-beam variable-layer optimization."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from layer_search import (
    LayerSearchConfig,
    ParetoBeamLayerSearch,
    TopologyCandidate,
    write_layer_search_results,
)
from lm_optimizer import BandSpec, LMThicknessOptimizer
from optimize_film import run as optimize_film_run
from optimize_film import load_stack_txt
from variable_layer_optimize import prepare_variable_layer_inputs


class LayerCountCalculator:
    """Deterministic spectrum whose target is reached with three layers."""

    def spectrum(self, layers, wavelengths, _theta0, **_kwargs):
        reflectance = min(1.0, 0.2 * len(layers))
        return (
            [reflectance for _ in wavelengths],
            [1.0 - reflectance for _ in wavelengths],
        )

    def _N(self, material):
        return {"h": 2.0, "l": 1.5}[material]


def make_optimizer() -> LMThicknessOptimizer:
    return LMThicknessOptimizer(
        LayerCountCalculator(),
        [BandSpec(500e-9, 600e-9, R_target=0.6)],
        method="trf",
        wavelength_step=20e-9,
        thickness_weight=0.0,
        max_iter=2,
        thickness_bounds={
            "h": (10e-9, 300e-9),
            "l": (10e-9, 300e-9),
        },
        multistart_seed=3,
    )


class LayerSearchTests(unittest.TestCase):
    def test_generated_seed_uses_search_midpoint_and_enables_layer_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = os.path.join(tmp, "config.json")
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "method": "multistart",
                        "stack_init": {
                            "mode": "lhl",
                            "design_wavelength_nm": 1000,
                        },
                        "layer_search": {
                            "min_layers": 4,
                            "max_layers": 10,
                        },
                        "bands": [
                            {
                                "wavelength_nm": [800, 1200],
                                "objective": "maximize",
                            }
                        ],
                        "output_dir": "result",
                    },
                    handle,
                )
            stack_path, effective_path = prepare_variable_layer_inputs(
                config_path=config_path
            )
            _incident, films, _substrate = load_stack_txt(stack_path)
            self.assertEqual(len(films), 7)
            self.assertEqual(films[0].material, "sio2")
            with open(effective_path, encoding="utf-8") as handle:
                effective = json.load(handle)
            self.assertTrue(effective["layer_search"]["enabled"])
            self.assertEqual(effective["layer_search"]["initial_layers"], 7)
            self.assertEqual(
                effective["generated_variable_layer_stack"]["max_layers"],
                10,
            )

    def test_search_selects_best_layer_count_and_restores_optimizer(self):
        optimizer = make_optimizer()
        original_method = optimizer.method
        original_wavelengths = list(optimizer.wavelengths)
        config = LayerSearchConfig(
            min_layers=1,
            max_layers=3,
            materials=("h", "l"),
            beam_width=4,
            offspring_per_parent=4,
            max_generations=2,
            stagnation_generations=1,
            coarse_optical_starts=1,
            coarse_sobol_starts=1,
            coarse_local_max_iter=1,
            pareto_archive_size=12,
            preserve_per_band=1,
            preserve_per_layer_count=1,
            final_top_k=3,
            initial_random_topologies=0,
            seed=3,
        )
        result = ParetoBeamLayerSearch(optimizer, config).run(
            [("h", 100e-9)], verbose=False
        )
        self.assertEqual(len(result.result.layers), 3)
        self.assertAlmostEqual(result.result.cost, 0.0, places=12)
        self.assertGreaterEqual(result.evaluated_topologies, 6)
        self.assertEqual(optimizer.method, original_method)
        self.assertEqual(optimizer.wavelengths, original_wavelengths)

    def test_mutations_include_layer_addition_and_deletion(self):
        optimizer = make_optimizer()
        config = LayerSearchConfig(
            min_layers=1,
            max_layers=4,
            materials=("h", "l"),
            offspring_per_parent=20,
        )
        search = ParetoBeamLayerSearch(optimizer, config)
        parent = TopologyCandidate(
            7,
            [("h", 100e-9), ("l", 120e-9)],
            generation=0,
            operation="input",
        )
        operations = [
            operation
            for bucket in search._mutation_buckets(parent, generation=1)
            for _, operation in bucket
        ]
        self.assertTrue(any(value.startswith("insert@") for value in operations))
        self.assertTrue(any(value.startswith("delete@") for value in operations))
        self.assertTrue(
            any(value.startswith("insert_pair@") for value in operations)
        )

    def test_config_validation_and_ranking_outputs(self):
        with self.assertRaisesRegex(ValueError, "min_layers"):
            LayerSearchConfig.from_mapping(
                {"min_layers": 4, "max_layers": 2},
                [("h", 100e-9)],
                default_materials=("h", "l"),
            )

        optimizer = make_optimizer()
        config = LayerSearchConfig(
            min_layers=1,
            max_layers=2,
            materials=("h", "l"),
            beam_width=2,
            offspring_per_parent=2,
            max_generations=1,
            coarse_optical_starts=1,
            coarse_sobol_starts=0,
            pareto_archive_size=6,
            preserve_per_band=1,
            preserve_per_layer_count=1,
            final_top_k=2,
            initial_random_topologies=0,
        )
        result = ParetoBeamLayerSearch(optimizer, config).run(
            [("h", 100e-9)], verbose=False
        )
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, json_path = write_layer_search_results(
                os.path.join(tmp, "ranking"), result
            )
            self.assertTrue(os.path.isfile(csv_path))
            with open(json_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(
                payload["evaluated_topologies"],
                result.evaluated_topologies,
            )
            self.assertEqual(len(payload["ranking"]), len(result.ranking))

    def test_optimize_film_entrypoint_writes_variable_layer_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            stack_path = os.path.join(tmp, "stack.txt")
            config_path = os.path.join(tmp, "config.json")
            output_dir = os.path.join(tmp, "out")
            with open(stack_path, "w", encoding="utf-8") as handle:
                handle.write(
                    "0 air 0 1.0 0\n"
                    "1 h 100 2.0 0\n"
                    "2 l 100 1.5 0\n"
                    "3 glass 0 1.52 0\n"
                )
            config = {
                "method": "trf",
                "max_iter": 1,
                "wavelength_step_nm": 50,
                "bands": [
                    {
                        "wavelength_nm": [500, 600],
                        "R_target": 0.5,
                        "objective": "maximize",
                    }
                ],
                "nk_source": "fixed",
                "thickness_bounds_nm": {
                    "h": [10, 300],
                    "l": [10, 300],
                },
                "layer_search": {
                    "enabled": True,
                    "materials": ["h", "l"],
                    "min_layers": 1,
                    "max_layers": 2,
                    "beam_width": 2,
                    "offspring_per_parent": 2,
                    "max_generations": 1,
                    "stagnation_generations": 1,
                    "coarse_optical_starts": 1,
                    "coarse_sobol_starts": 0,
                    "coarse_local_max_iter": 1,
                    "pareto_archive_size": 4,
                    "preserve_per_band": 1,
                    "preserve_per_layer_count": 1,
                    "final_top_k": 1,
                    "initial_random_topologies": 0,
                },
                "checkpoint_on_best": False,
                "plot_wavelength_nm": [500, 600],
                "plot_step_nm": 50,
                "output_dir": output_dir,
            }
            with open(config_path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)
            with (
                patch("optimize_film.plot_results"),
                patch("optimize_film.plot_rt"),
                patch("optimize_film.plot_used_nk"),
                redirect_stdout(StringIO()),
            ):
                exit_code = optimize_film_run(stack_path, config_path)
            self.assertEqual(exit_code, 0)
            self.assertTrue(
                os.path.isfile(
                    os.path.join(output_dir, "layer_search_ranking.csv")
                )
            )
            self.assertTrue(
                os.path.isfile(os.path.join(output_dir, "stack_best.txt"))
            )
            candidate_dir = os.path.join(
                output_dir, "layer_search_candidates"
            )
            candidate_files = os.listdir(candidate_dir)
            self.assertTrue(
                any(name.endswith("_spectrum.csv") for name in candidate_files)
            )
            self.assertTrue(
                any(name.endswith("_band_stats.csv") for name in candidate_files)
            )
            self.assertTrue(
                any(name.endswith("_stack.txt") for name in candidate_files)
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(
                        output_dir, "layer_search_candidates_rt.png"
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
