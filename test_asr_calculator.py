"""
Tests for asr_calculator.py

Uses plain unittest + assert statements. Run with:
    python test_asr_calculator.py
"""

import csv
import os
import tempfile
import unittest

import numpy as np
from scipy.stats import chi2, norm

import asr_calculator as asr


class TestDirectStandardization(unittest.TestCase):
    def setUp(self):
        # Hand-computable case: 3 age bands, all age-specific rates equal
        # to 0.005/person-year, so ASR must equal 0.005 exactly regardless
        # of weights (weighted average of a constant is the constant).
        self.weights = np.array([0.5, 0.3, 0.2])
        self.counts = np.array([50.0, 30.0, 20.0])
        self.person_years = np.array([10000.0, 6000.0, 4000.0])

    def test_asr_matches_hand_calculation(self):
        asr_val, var_val = asr.direct_standardize(
            self.weights, self.counts, self.person_years
        )
        self.assertAlmostEqual(asr_val, 0.005, places=12)
        # variance = sum(w_i^2 * d_i / n_i^2), computed by hand:
        # term1 = 0.25*50/1e8 = 1.25e-7
        # term2 = 0.09*30/3.6e7 = 7.5e-8
        # term3 = 0.04*20/1.6e7 = 5.0e-8
        # total = 2.5e-7
        self.assertAlmostEqual(var_val, 2.5e-7, places=15)

    def test_fay_feuer_ci_matches_formula(self):
        asr_val, var_val = asr.direct_standardize(
            self.weights, self.counts, self.person_years
        )
        alpha = 0.05
        lower, upper = asr.fay_feuer_ci(
            asr_val, var_val, self.weights, self.person_years, alpha
        )

        # Independently recompute via the Fay & Feuer (1997) formula
        # straight from the docstring, as a regression check against the
        # implementation.
        wm = float(np.max(self.weights / self.person_years))
        df_l = 2.0 * asr_val ** 2 / var_val
        expected_lower = (var_val / (2.0 * asr_val)) * chi2.ppf(alpha / 2.0, df_l)
        df_u = 2.0 * (asr_val + wm) ** 2 / (var_val + wm ** 2)
        expected_upper = ((var_val + wm ** 2) / (2.0 * (asr_val + wm))) * chi2.ppf(
            1.0 - alpha / 2.0, df_u
        )

        self.assertAlmostEqual(lower, expected_lower, places=15)
        self.assertAlmostEqual(upper, expected_upper, places=15)
        # Sanity: CI must bracket the point estimate.
        self.assertLess(lower, asr_val)
        self.assertLess(asr_val, upper)

    def test_ci_zero_events_gives_zero_lower_bound(self):
        weights = np.array([0.5, 0.5])
        counts = np.array([0.0, 0.0])
        person_years = np.array([1000.0, 1000.0])
        asr_val, var_val = asr.direct_standardize(weights, counts, person_years)
        self.assertEqual(asr_val, 0.0)
        lower, upper = asr.fay_feuer_ci(asr_val, var_val, weights, person_years)
        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, 0.0)


class TestIndirectStandardizationSMR(unittest.TestCase):
    def test_smr_matches_hand_calculation(self):
        # Reference rate 0.002/person-year applied to 5000 person-years
        # of study data -> expected = 10 events. Observed = 20 -> SMR = 2.0
        study = asr.AgeSpecificData(
            age_groups=["0-4", "5-9"],
            counts=[15, 5],
            person_years=[2500, 2500],
        )
        ref_rates = {"0-4": 0.002, "5-9": 0.002}
        observed, expected, smr = asr.indirect_standardize(study, ref_rates)
        self.assertEqual(observed, 20.0)
        self.assertAlmostEqual(expected, 10.0, places=10)
        self.assertAlmostEqual(smr, 2.0, places=10)

    def test_smr_ci_matches_poisson_formula(self):
        observed, expected = 20.0, 10.0
        alpha = 0.05
        lower, upper = asr.smr_poisson_ci(observed, expected, alpha)
        expected_lower = chi2.ppf(alpha / 2.0, 2 * observed) / 2.0 / expected
        expected_upper = chi2.ppf(1.0 - alpha / 2.0, 2 * (observed + 1)) / 2.0 / expected
        self.assertAlmostEqual(lower, expected_lower, places=12)
        self.assertAlmostEqual(upper, expected_upper, places=12)
        self.assertLess(lower, 2.0)
        self.assertLess(2.0, upper)

    def test_smr_zero_observed_gives_zero_lower_bound(self):
        study = asr.AgeSpecificData(["0-4"], [0], [1000])
        ref_rates = {"0-4": 0.01}
        observed, expected, smr = asr.indirect_standardize(study, ref_rates)
        self.assertEqual(smr, 0.0)
        lower, upper = asr.smr_poisson_ci(observed, expected)
        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, 0.0)

    def test_reference_rates_from_counts(self):
        ref_data = asr.AgeSpecificData(["0-4", "5-9"], [20, 10], [10000, 5000])
        rates = asr.reference_rates_from_counts(ref_data)
        self.assertAlmostEqual(rates["0-4"], 0.002, places=10)
        self.assertAlmostEqual(rates["5-9"], 0.002, places=10)


class TestRateRatio(unittest.TestCase):
    def test_ratio_and_ci_matches_formula(self):
        asr1, var1 = 0.01, 1e-6
        asr2, var2 = 0.005, 5e-7
        alpha = 0.05
        ratio, lower, upper = asr.rate_ratio_ci(asr1, var1, asr2, var2, alpha)
        self.assertAlmostEqual(ratio, 2.0, places=10)

        se_log = np.sqrt(var1 / asr1 ** 2 + var2 / asr2 ** 2)
        z = norm.ppf(1 - alpha / 2)
        expected_lower = ratio * np.exp(-z * se_log)
        expected_upper = ratio * np.exp(z * se_log)
        self.assertAlmostEqual(lower, expected_lower, places=10)
        self.assertAlmostEqual(upper, expected_upper, places=10)
        self.assertLess(lower, ratio)
        self.assertLess(ratio, upper)

    def test_identical_rates_give_ratio_of_one(self):
        ratio, lower, upper = asr.rate_ratio_ci(0.02, 1e-6, 0.02, 1e-6)
        self.assertAlmostEqual(ratio, 1.0, places=10)
        self.assertLess(lower, 1.0)
        self.assertGreater(upper, 1.0)


class TestStandardPopulationsAndAlignment(unittest.TestCase):
    def test_builtin_standards_normalize_to_one(self):
        for name in ("who2000", "us2000"):
            age_groups, weights = asr.standard_weights(asr.BUILTIN_STANDARDS[name])
            self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=10)
            self.assertEqual(len(age_groups), len(set(age_groups)))

    def test_align_reorders_to_standard_order(self):
        data = asr.AgeSpecificData(
            age_groups=["5-9", "0-4"],
            counts=[30, 50],
            person_years=[6000, 10000],
        )
        std_age_groups = ["0-4", "5-9"]
        std_weights = np.array([0.6, 0.4])
        w, counts, py = asr.align_to_standard(data, std_age_groups, std_weights)
        np.testing.assert_array_equal(counts, [50, 30])
        np.testing.assert_array_equal(py, [10000, 6000])
        np.testing.assert_array_equal(w, [0.6, 0.4])

    def test_align_raises_on_missing_age_band(self):
        data = asr.AgeSpecificData(["0-4"], [50], [10000])
        with self.assertRaises(ValueError):
            asr.align_to_standard(data, ["0-4", "5-9"], np.array([0.6, 0.4]))

    def test_align_raises_on_extra_age_band(self):
        data = asr.AgeSpecificData(["0-4", "5-9", "10-14"], [50, 30, 5], [10000, 6000, 2000])
        with self.assertRaises(ValueError):
            asr.align_to_standard(data, ["0-4", "5-9"], np.array([0.6, 0.4]))


class TestCsvIOAndEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

        self.data_path = os.path.join(self.tmpdir, "pop.csv")
        with open(self.data_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["age_group", "count", "person_years"])
            writer.writerow(["0-4", "50", "10000"])
            writer.writerow(["5-9", "30", "6000"])
            writer.writerow(["10-14", "20", "4000"])

        self.std_path = os.path.join(self.tmpdir, "std.csv")
        with open(self.std_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["age_group", "population"])
            writer.writerow(["0-4", "500"])
            writer.writerow(["5-9", "300"])
            writer.writerow(["10-14", "200"])

    def test_read_and_end_to_end_direct_report(self):
        data = asr.read_age_specific_csv(self.data_path)
        std_rows = asr.load_standard_population(self.std_path)
        result = asr.direct_standardization_report(data, std_rows, alpha=0.05, per=1000.0)

        # All age-specific rates are 0.005/person-year in this fixture, so
        # the standardized rate must equal the crude rate exactly (5.0 per
        # 1000 person-years) regardless of weighting.
        self.assertAlmostEqual(result["crude_rate"], 5.0, places=10)
        self.assertAlmostEqual(result["asr"], 5.0, places=10)
        self.assertLess(result["asr_lower"], result["asr"])
        self.assertGreater(result["asr_upper"], result["asr"])

    def test_missing_column_raises(self):
        bad_path = os.path.join(self.tmpdir, "bad.csv")
        with open(bad_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["age_group", "count"])
            writer.writerow(["0-4", "50"])
        with self.assertRaises(ValueError):
            asr.read_age_specific_csv(bad_path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
