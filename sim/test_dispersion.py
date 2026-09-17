import unittest

import dispersion


class TFCalcMaterialTests(unittest.TestCase):
    def test_supplied_tfcalc_points(self):
        tables = {
            "tfcalc-sio2": (
                (310, 1.485, 0),
                (388, 1.472, 0),
                (400, 1.470, 0),
                (496, 1.463, 0),
                (564, 1.459, 0),
                (653, 1.457, 0),
                (775, 1.454, 0),
                (954, 1.451, 0),
                (1240, 1.448, 0),
                (1550, 1.444, 0),
                (2066, 1.437, 0),
            ),
            "tfcalc-tio2-a": (
                (300, 2.600, 0.010),
                (400, 2.520, 0),
                (500, 2.420, 0),
                (600, 2.350, 0),
                (700, 2.330, 0),
                (800, 2.300, 0),
                (1800, 2.300, 0),
            ),
            "tfcalc-tio2-rutile": (
                (300, 3.500, 3.200),
                (400, 3.200, 0),
                (500, 3.000, 0),
                (600, 2.900, 0),
                (700, 2.830, 0),
                (800, 2.790, 0),
                (1800, 2.700, 0),
            ),
        }
        for material, rows in tables.items():
            for wavelength_nm, expected_n, expected_k in rows:
                with self.subTest(material=material, wavelength_nm=wavelength_nm):
                    actual = dispersion.material_n(material, wavelength_nm * 1e-9)
                    self.assertAlmostEqual(actual.real, expected_n, places=12)
                    self.assertAlmostEqual(actual.imag, expected_k, places=12)

    def test_existing_defaults_are_unchanged(self):
        wavelength = 550e-9
        self.assertEqual(dispersion.material_n("sio2", wavelength), dispersion.sio2(wavelength))
        self.assertEqual(dispersion.material_n("tio2", wavelength), dispersion.tio2_pvd(wavelength))


if __name__ == "__main__":
    unittest.main()
