import unittest

from plot_rt_txt import validate_windows, window_average_stats


class WindowStatsTests(unittest.TestCase):
    def test_multiple_window_averages_include_closed_endpoints(self):
        wavelengths = [value * 1e-9 for value in (300, 400, 500, 600)]
        rows = window_average_stats(
            wavelengths,
            [0.1, 0.2, 0.3, 0.4],
            [0.8, 0.7, 0.6, 0.5],
            [(300, 400), (500, 600)],
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["sample_count"], 2)
        self.assertAlmostEqual(rows[0]["R_mean"], 0.15)
        self.assertAlmostEqual(rows[0]["T_mean"], 0.75)
        self.assertAlmostEqual(rows[1]["R_mean"], 0.35)
        self.assertAlmostEqual(rows[1]["T_mean"], 0.55)

    def test_no_windows_is_the_default(self):
        self.assertEqual(validate_windows(None, 300, 1800), [])
        self.assertEqual(window_average_stats([300e-9], [0.1], [0.9], []), [])

    def test_rejects_invalid_or_out_of_range_windows(self):
        with self.assertRaises(SystemExit):
            validate_windows([(500, 500)], 300, 1800)
        with self.assertRaises(SystemExit):
            validate_windows([(200, 500)], 300, 1800)


if __name__ == "__main__":
    unittest.main()
