"""Focused regression tests for optimizer objective semantics and TRF."""

from __future__ import annotations

import unittest

from lm_optimizer import BandSpec, LMThicknessOptimizer, build_residuals


class LinearReflectanceCalculator:
    """Small deterministic stand-in for TMM: R=d/500 nm, T=1-R."""

    def spectrum(self, layers, wavelengths, _theta0, **_kwargs):
        reflectance = max(0.0, min(1.0, layers[0][1] / 500e-9))
        return (
            [reflectance for _ in wavelengths],
            [1.0 - reflectance for _ in wavelengths],
        )


class ResidualTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
