"""Focused regression tests for optimizer objective semantics and TRF."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from lm_optimizer import (
    BandSpec,
    LMThicknessOptimizer,
    build_residuals,
    is_better_checkpoint,
    parse_multistart_sampling_bounds_nm,
    parse_thickness_bounds_nm,
)
from multistart_optimize import prepare_inputs
from optimize_film import load_stack_txt


class LinearReflectanceCalculator:
    """Small deterministic stand-in for TMM: R=d/500 nm, T=1-R."""

    def spectrum(self, layers, wavelengths, _theta0, **_kwargs):
        reflectance = max(0.0, min(1.0, layers[0][1] / 500e-9))
        return (
            [reflectance for _ in wavelengths],
            [1.0 - reflectance for _ in wavelengths],
        )


class FixedIndexCalculator(LinearReflectanceCalculator):
    """Linear test spectrum plus known indices for optical-QW sampling."""

    def _N(self, material):
        return {"h": 2.0, "l": 1.5}[material]


class ResidualTests(unittest.TestCase):
    def test_checkpoint_selection_uses_loss_only(self):
        self.assertTrue(is_better_checkpoint(0.1, 0.2))
        self.assertFalse(is_better_checkpoint(0.2, 0.1))
        self.assertFalse(is_better_checkpoint(0.1, 0.1))

    def test_all_inequality_directions_are_present(self):
        band = BandSpec(
            400e-9,
            700e-9,
            R_min=0.5,
            R_max=0.7,
            T_min=0.2,
            T_max=0.4,
        )
        residuals = build_residuals(
            [],
            [band],
            [550e-9],
            [0.8],
            [0.1],
            thickness_weight=0.0,
        )
        # R_min and T_max are satisfied; R_max and T_min each miss by 0.1.
        self.assertEqual(len(residuals), 4)
        self.assertAlmostEqual(residuals[0], 0.0)
        self.assertGreater(residuals[1], 0.0)
        self.assertGreater(residuals[2], 0.0)
        self.assertAlmostEqual(residuals[3], 0.0)

    def test_residual_length_does_not_change_at_feasibility_boundary(self):
        band = BandSpec(400e-9, 700e-9, R_min=0.5, T_max=0.5)
        feasible = build_residuals(
            [("x", 100e-9)], [band], [550e-9], [0.6], [0.4]
        )
        infeasible = build_residuals(
            [("x", 100e-9)], [band], [550e-9], [0.4], [0.6]
        )
        self.assertEqual(len(feasible), len(infeasible))


class TRFTests(unittest.TestCase):
    def test_hard_total_thickness_cap_projects_all_free_layers(self):
        optimizer = LMThicknessOptimizer(
            LinearReflectanceCalculator(),
            [BandSpec(500e-9, 600e-9, R_target=0.4)],
            max_total_thickness=250e-9,
        )
        projected = optimizer._project(
            ["x", "x"],
            [200e-9, 200e-9],
            [0, 1],
        )
        self.assertAlmostEqual(sum(projected) * 1e9, 250.0, places=6)
        self.assertAlmostEqual(projected[0] * 1e9, 125.0, places=6)
        self.assertAlmostEqual(projected[1] * 1e9, 125.0, places=6)

    def test_hard_total_thickness_cap_preserves_fixed_layers(self):
        optimizer = LMThicknessOptimizer(
            LinearReflectanceCalculator(),
            [BandSpec(500e-9, 600e-9, R_target=0.4)],
            max_total_thickness=250e-9,
        )
        projected = optimizer._project(
            ["x", "x"],
            [150e-9, 150e-9],
            [0],
        )
        self.assertAlmostEqual(projected[0] * 1e9, 100.0, places=6)
        self.assertAlmostEqual(projected[1] * 1e9, 150.0, places=6)

    def test_infeasible_total_thickness_cap_is_rejected(self):
        optimizer = LMThicknessOptimizer(
            LinearReflectanceCalculator(),
            [BandSpec(500e-9, 600e-9, R_target=0.4)],
            max_total_thickness=15e-9,
        )
        with self.assertRaisesRegex(ValueError, "need at least 20.000 nm"):
            optimizer._project(["x", "x"], [100e-9, 100e-9], [0, 1])

    def test_trf_result_respects_hard_total_thickness_cap(self):
        optimizer = LMThicknessOptimizer(
            LinearReflectanceCalculator(),
            [BandSpec(500e-9, 600e-9, R_target=0.8)],
            method="trf",
            wavelength_step=20e-9,
            thickness_weight=0.0,
            max_iter=30,
            max_total_thickness=250e-9,
        )
        result = optimizer.optimize(
            [("x", 200e-9), ("x", 200e-9)],
            verbose=False,
        )
        self.assertLessEqual(
            sum(d for _, d in result.layers),
            250e-9 + 1e-15,
        )
        self.assertAlmostEqual(result.layers[0][1] * 1e9, 240.0, places=3)
        self.assertAlmostEqual(result.layers[1][1] * 1e9, 10.0, places=3)

    def test_json_thickness_bounds_constrain_multistart_and_local_search(self):
        bounds = parse_thickness_bounds_nm({"x": [100, 150]})
        optimizer = LMThicknessOptimizer(
            LinearReflectanceCalculator(),
            [BandSpec(500e-9, 600e-9, R_target=0.8)],
            method="multistart",
            multistart_n=4,
            multistart_seed=2,
            wavelength_step=20e-9,
            thickness_weight=0.0,
            max_iter=20,
            min_thickness=8e-9,
            thickness_bounds=bounds,
        )
        result = optimizer.optimize([("x", 120e-9)], verbose=False)
        self.assertGreaterEqual(result.layers[0][1], 100e-9)
        self.assertLessEqual(result.layers[0][1], 150e-9 + 1e-15)
        self.assertAlmostEqual(result.layers[0][1] * 1e9, 150.0, places=3)

    def test_multistart_sampling_bounds_do_not_limit_local_search(self):
        hard_bounds = parse_thickness_bounds_nm({"x": [50, 450]})
        sampling_bounds = parse_multistart_sampling_bounds_nm({"x": [100, 150]})
        optimizer = LMThicknessOptimizer(
            LinearReflectanceCalculator(),
            [BandSpec(500e-9, 600e-9, R_target=0.8)],
            method="multistart",
            multistart_n=4,
            multistart_seed=2,
            wavelength_step=20e-9,
            thickness_weight=0.0,
            max_iter=20,
            thickness_bounds=hard_bounds,
            multistart_sampling_bounds=sampling_bounds,
        )
        sample_lo, sample_hi = optimizer.multistart_bounds_for("x")
        self.assertAlmostEqual(sample_lo * 1e9, 100.0)
        self.assertAlmostEqual(sample_hi * 1e9, 150.0)
        result = optimizer.optimize([("x", 120e-9)], verbose=False)
        self.assertAlmostEqual(result.layers[0][1] * 1e9, 400.0, places=3)

    def test_sampling_bounds_must_be_inside_thickness_bounds(self):
        with self.assertRaisesRegex(ValueError, "must be inside"):
            LMThicknessOptimizer(
                LinearReflectanceCalculator(),
                [BandSpec(500e-9, 600e-9, R_target=0.4)],
                thickness_bounds=parse_thickness_bounds_nm({"x": [50, 200]}),
                multistart_sampling_bounds=parse_multistart_sampling_bounds_nm(
                    {"x": [40, 150]}
                ),
            )

    def test_sobol_multistart_sampling_is_bounded_and_reproducible(self):
        kwargs = dict(
            method="multistart",
            multistart_n=8,
            multistart_seed=5,
            multistart_sampler="sobol",
            thickness_bounds=parse_thickness_bounds_nm({"x": [50, 450]}),
            multistart_sampling_bounds=parse_multistart_sampling_bounds_nm(
                {"x": [100, 150]}
            ),
        )
        optimizer = LMThicknessOptimizer(
            LinearReflectanceCalculator(),
            [BandSpec(500e-9, 600e-9, R_target=0.4)],
            **kwargs,
        )
        starts = optimizer._generate_multistart_starts(
            ["x"], [120e-9], [0], verbose=False
        )
        repeated = optimizer._generate_multistart_starts(
            ["x"], [120e-9], [0], verbose=False
        )
        self.assertEqual(starts, repeated)
        self.assertEqual(len(starts), 8)
        for start in starts[1:]:
            self.assertGreaterEqual(start[0], 100e-9)
            self.assertLessEqual(start[0], 150e-9)

    def test_multistart_samples_directly_inside_total_thickness_cap(self):
        for sampler in ("lhs", "sobol", "optical_qw"):
            with self.subTest(sampler=sampler):
                optimizer = LMThicknessOptimizer(
                    LinearReflectanceCalculator(),
                    [BandSpec(500e-9, 600e-9, R_target=0.4)],
                    method="multistart",
                    multistart_n=32,
                    multistart_seed=7,
                    multistart_sampler=sampler,
                    thickness_bounds=parse_thickness_bounds_nm(
                        {"x": [10, 200]}
                    ),
                    max_total_thickness=250e-9,
                )
                starts = optimizer._generate_multistart_starts(
                    ["x", "x", "x"],
                    [50e-9, 50e-9, 50e-9],
                    [0, 1, 2],
                    verbose=False,
                )
                random_starts = starts[1:]
                self.assertTrue(
                    all(sum(start) <= 250e-9 + 1e-15 for start in random_starts)
                )
                self.assertTrue(
                    all(
                        10e-9 - 1e-15 <= d <= 200e-9 + 1e-15
                        for start in random_starts
                        for d in start
                    )
                )
                # Direct feasible sampling should not collapse all infeasible
                # box draws onto the total-thickness boundary.
                self.assertTrue(
                    any(sum(start) < 240e-9 for start in random_starts)
                )

    def test_optical_qw_sampler_maps_shared_pair_wavelengths_to_thickness(self):
        optimizer = LMThicknessOptimizer(
            FixedIndexCalculator(),
            [BandSpec(500e-9, 1500e-9, R_target=0.9)],
            method="multistart",
            multistart_n=8,
            multistart_seed=11,
            multistart_sampler="optical_qw",
            optical_q_range=(0.999999, 1.000001),
            optical_wavelength_range=(500e-9, 1500e-9),
            optical_pair_shared_wavelength=True,
            optical_chirp=True,
            optical_sampler_fraction=1.0,
            thickness_bounds={
                "h": (10e-9, 500e-9),
                "l": (10e-9, 500e-9),
            },
        )
        starts = optimizer._generate_multistart_starts(
            ["h", "l", "h", "l"],
            [100e-9, 100e-9, 100e-9, 100e-9],
            [0, 1, 2, 3],
            verbose=False,
        )
        repeated = optimizer._generate_multistart_starts(
            ["h", "l", "h", "l"],
            [100e-9, 100e-9, 100e-9, 100e-9],
            [0, 1, 2, 3],
            verbose=False,
        )
        self.assertEqual(starts, repeated)
        for start in starts[1:]:
            first_h = 4.0 * 2.0 * start[0]
            first_l = 4.0 * 1.5 * start[1]
            second_h = 4.0 * 2.0 * start[2]
            second_l = 4.0 * 1.5 * start[3]
            self.assertAlmostEqual(first_h / first_l, 1.0, places=5)
            self.assertAlmostEqual(second_h / second_l, 1.0, places=5)
            self.assertLess(first_h, second_h)

    def test_total_cap_sampling_counts_fixed_layer_thickness(self):
        optimizer = LMThicknessOptimizer(
            LinearReflectanceCalculator(),
            [BandSpec(500e-9, 600e-9, R_target=0.4)],
            method="multistart",
            multistart_n=16,
            multistart_seed=3,
            multistart_sampler="sobol",
            thickness_bounds=parse_thickness_bounds_nm({"x": [10, 200]}),
            max_total_thickness=260e-9,
        )
        starts = optimizer._generate_multistart_starts(
            ["x", "x", "x"],
            [100e-9, 50e-9, 50e-9],
            [1, 2],
            verbose=False,
        )
        self.assertTrue(all(sum(start) <= 260e-9 + 1e-15 for start in starts))
        self.assertTrue(all(abs(start[0] - 100e-9) <= 1e-15 for start in starts))

    def test_extra_trees_sampler_selects_requested_start_count(self):
        optimizer = LMThicknessOptimizer(
            LinearReflectanceCalculator(),
            [BandSpec(500e-9, 600e-9, R_target=0.4)],
            method="multistart",
            multistart_n=4,
            multistart_seed=5,
            multistart_sampler="extra_trees",
            multistart_candidate_n=16,
            surrogate_trees=10,
            surrogate_pool_n=32,
            thickness_weight=0.0,
            thickness_bounds=parse_thickness_bounds_nm({"x": [50, 450]}),
            multistart_sampling_bounds=parse_multistart_sampling_bounds_nm(
                {"x": [100, 300]}
            ),
            max_total_thickness=350e-9,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            starts = optimizer._generate_multistart_starts(
                ["x", "x"], [120e-9, 120e-9], [0, 1], verbose=True
            )
        self.assertEqual(len(starts), 4)
        self.assertEqual(starts[0], [120e-9, 120e-9])
        self.assertTrue(all(sum(start) <= 350e-9 + 1e-15 for start in starts))
        log = output.getvalue()
        self.assertIn(
            "surrogate initial prescreen (real TMM loss): started", log
        )
        self.assertIn("surrogate training final (10 Extra Trees, n=16)", log)
        self.assertIn("surrogate inference final (pool=32): completed", log)
        self.assertIn("surrogate full-grid validation: completed", log)
        self.assertIn("preserved=", log)

    def test_surrogate_preserves_overall_band_and_thin_specialists(self):
        optimizer = LMThicknessOptimizer(
            LinearReflectanceCalculator(),
            [
                BandSpec(500e-9, 600e-9, R_target=0.2),
                BandSpec(700e-9, 800e-9, R_target=0.8),
            ],
            surrogate_band_candidates_per_band=1,
            surrogate_preserve_thinnest=True,
        )
        selected = optimizer._preserved_specialist_indices(
            costs=[0.4, 0.3, 0.2, 0.1],
            band_merits=[
                [0.5, 0.1],
                [0.1, 0.4],
                [0.3, 0.05],
                [0.2, 0.2],
            ],
            total_thicknesses=[400e-9, 300e-9, 200e-9, 100e-9],
            limit=4,
        )
        self.assertEqual(selected, [3, 1, 2])

    def test_optical_extra_trees_uses_hybrid_training_and_pool(self):
        optimizer = LMThicknessOptimizer(
            FixedIndexCalculator(),
            [BandSpec(500e-9, 1500e-9, R_target=0.9)],
            method="multistart",
            multistart_n=4,
            multistart_seed=13,
            multistart_sampler="optical_extra_trees",
            multistart_candidate_n=16,
            surrogate_trees=10,
            surrogate_pool_n=32,
            surrogate_rounds=2,
            surrogate_initial_n=8,
            surrogate_batch_n=4,
            surrogate_wavelength_step=50e-9,
            surrogate_validation_factor=2.0,
            surrogate_elite_fraction=0.25,
            surrogate_elite_jitter=0.1,
            optical_q_range=(0.7, 1.3),
            optical_wavelength_range=(800e-9, 1500e-9),
            optical_sampler_fraction=0.6,
            thickness_bounds={
                "h": (10e-9, 500e-9),
                "l": (10e-9, 500e-9),
            },
            max_total_thickness=700e-9,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            starts = optimizer._generate_multistart_starts(
                ["h", "l", "h", "l"],
                [100e-9, 100e-9, 100e-9, 100e-9],
                [0, 1, 2, 3],
                verbose=True,
            )
        self.assertEqual(len(starts), 4)
        self.assertTrue(all(sum(start) <= 700e-9 + 1e-15 for start in starts))
        log = output.getvalue()
        self.assertIn(
            "surrogate candidate pool (optical-QW/Sobol n=32): completed",
            log,
        )
        self.assertIn("surrogate active round 1/2: completed", log)
        self.assertIn("surrogate active round 2/2: completed", log)
        self.assertIn("training=16  rounds=2", log)
        self.assertIn("surrogate full-grid validation: completed", log)
        self.assertIn("source=optical-QW/Sobol", log)

    def test_incremental_surrogate_selection_matches_brute_force(self):
        import numpy as np

        optimizer = LMThicknessOptimizer(
            LinearReflectanceCalculator(),
            [BandSpec(500e-9, 600e-9, R_target=0.4)],
            surrogate_diversity_weight=0.2,
        )
        pool = np.asarray(
            [
                [0.1, 0.2],
                [0.8, 0.7],
                [0.4, 0.9],
                [0.6, 0.1],
                [0.3, 0.5],
            ],
            dtype=np.float64,
        )
        score = np.asarray([0.4, 0.1, 0.3, 0.2, 0.35])
        initial = np.asarray([0.2, 0.2])
        available = set(range(len(pool)))
        selected_x = [initial]
        expected = []
        while len(expected) + 1 < 4:
            index = min(
                available,
                key=lambda i: float(score[i])
                - 0.2
                * min(
                    float(np.sqrt(np.mean((pool[i] - chosen) ** 2)))
                    for chosen in selected_x
                ),
            )
            available.remove(index)
            expected.append(index)
            selected_x.append(pool[index])

        actual = optimizer._select_surrogate_pool_indices(
            pool,
            score,
            initial,
            4,
            score,
            verbose=False,
        )
        self.assertEqual(actual, expected)

    def test_invalid_json_thickness_bounds_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "min_nm < max_nm"):
            parse_thickness_bounds_nm({"tio2": [200, 100]})

    def test_duplicate_multistart_gpu_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            LMThicknessOptimizer(
                LinearReflectanceCalculator(),
                [BandSpec(500e-9, 600e-9, R_target=0.4)],
                multistart_gpu_ids=[0, 0],
            )

    def test_multistart_supports_minibatch_adam(self):
        optimizer = LMThicknessOptimizer(
            LinearReflectanceCalculator(),
            [BandSpec(500e-9, 600e-9, R_target=0.4)],
            method="multistart",
            multistart_method="adam",
            multistart_n=2,
            multistart_seed=3,
            wavelength_step=20e-9,
            thickness_weight=0.0,
            max_iter=2,
            mini_batch=True,
            batch_size=2,
            n_batches=1,
            n_epochs=2,
            shuffle_seed=4,
            multistart_final_polish_method="trf",
        )
        full_wavelengths = list(optimizer.wavelengths)
        result = optimizer.optimize([("x", 100e-9)], verbose=False)
        self.assertIn("multistart(adam", result.message)
        self.assertTrue(result.message.endswith("->trf"))
        self.assertEqual(optimizer.wavelengths, full_wavelengths)
        self.assertLess(result.cost, result.start_cost)

    def test_trf_finds_bounded_target_thickness(self):
        optimizer = LMThicknessOptimizer(
            LinearReflectanceCalculator(),
            [BandSpec(500e-9, 600e-9, R_target=0.4)],
            method="trf",
            wavelength_step=20e-9,
            thickness_weight=0.0,
            fd_step=0.5e-9,
            max_iter=30,
            min_thickness=10e-9,
        )
        result = optimizer.optimize([("x", 100e-9)], verbose=False)
        self.assertTrue(result.success)
        self.assertLess(result.cost, 1e-10)
        self.assertAlmostEqual(result.layers[0][1] * 1e9, 200.0, places=3)

    def test_auto_pipeline_can_run_without_global_fallback(self):
        optimizer = LMThicknessOptimizer(
            LinearReflectanceCalculator(),
            [BandSpec(500e-9, 600e-9, R_target=0.6, R_min=0.59)],
            method="auto",
            multistart_n=3,
            multistart_method="trf",
            multistart_seed=7,
            auto_de_fallback=False,
            wavelength_step=20e-9,
            thickness_weight=0.0,
            max_iter=20,
        )
        result = optimizer.optimize([("x", 80e-9)], verbose=False)
        self.assertTrue(result.success)
        self.assertIn("auto:multistart", result.message)
        self.assertAlmostEqual(result.layers[0][1] * 1e9, 300.0, places=3)

    def test_auto_pipeline_uses_de_when_constraints_remain_infeasible(self):
        optimizer = LMThicknessOptimizer(
            LinearReflectanceCalculator(),
            [BandSpec(500e-9, 600e-9, R_target=1.0, R_min=1.1)],
            method="auto",
            multistart_n=1,
            multistart_method="trf",
            auto_de_fallback=True,
            de_popsize=5,
            global_seed=3,
            wavelength_step=20e-9,
            thickness_weight=0.0,
            max_iter=2,
        )
        result = optimizer.optimize([("x", 80e-9)], verbose=False)
        self.assertIn("+de->", result.message)
        self.assertGreater(result.layers[0][1], 400e-9)


class GeneratedStackTests(unittest.TestCase):
    def test_prepare_inputs_generates_requested_layer_count(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = os.path.join(directory, "config.json")
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "stack_init": {
                            "mode": "lhl",
                            "design_wavelength_nm": 900,
                        },
                        "bands": [
                            {
                                "wavelength_nm": [800, 1000],
                                "objective": "maximize",
                            }
                        ],
                        "output_dir": "result",
                    },
                    fh,
                )
            stack_path, effective_path = prepare_inputs(
                n_layers=7, config_path=config_path
            )
            _incident, films, _substrate = load_stack_txt(stack_path)
            self.assertEqual(len(films), 7)
            self.assertEqual(films[0].material, "sio2")
            with open(effective_path, encoding="utf-8") as fh:
                effective = json.load(fh)
            self.assertEqual(effective["method"], "multistart")
            self.assertEqual(effective["generated_stack"]["layers"], 7)

            _, auto_config_path = prepare_inputs(
                n_layers=7,
                config_path=config_path,
                method_override="auto",
            )
            with open(auto_config_path, encoding="utf-8") as fh:
                auto_config = json.load(fh)
            self.assertEqual(auto_config["method"], "auto")


if __name__ == "__main__":
    unittest.main()
