#!/usr/bin/env python3
"""
Unit Test Suite for Age-Standardized Rate (ASR) Calculator
==========================================================
Pure Python unit tests verifying:
- Direct standardization and variance estimation
- Fay & Feuer (1997) gamma-distributed confidence intervals
- Exact Poisson confidence intervals for SMR / SIR
- Standardized Rate Ratio (SRR) and Rate Difference (SRD)
- Multi-standard population handling (WHO, US, Segi, European)
- Cumulative rate and risk metrics (0-74)
- CSV loading, batch processing, and CLI interfaces.
"""

import csv
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if ROOT_DIR.name == "tests":
    ROOT_DIR = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import asr_calculator as asr
from asr_calculator import (
    AgeSpecificData,
    ASRCalculator,
    BUILTIN_STANDARDS,
    direct_standardize,
    direct_standardization_report,
    fay_feuer_ci,
    indirect_standardize,
    smr_poisson_ci,
    read_age_specific_csv,
    normal_cdf,
    normal_ppf,
    chi2_ppf,
    standard_weights,
    align_to_standard,
)
import cli


class TestDirectStandardization(unittest.TestCase):
    def setUp(self):
        # 3 age bands, all age-specific rates equal to 0.005/person-year
        self.weights = [0.5, 0.3, 0.2]
        self.counts = [50.0, 30.0, 20.0]
        self.person_years = [10000.0, 6000.0, 4000.0]

    def test_asr_matches_hand_calculation(self):
        asr_val, var_val = direct_standardize(
            self.weights, self.counts, self.person_years
        )
        self.assertAlmostEqual(asr_val, 0.005, places=12)
        # variance = sum(w_i^2 * d_i / n_i^2) = 2.5e-7
        self.assertAlmostEqual(var_val, 2.5e-7, places=15)

    def test_fay_feuer_ci_brackets_estimate(self):
        asr_val, var_val = direct_standardize(
            self.weights, self.counts, self.person_years
        )
        alpha = 0.05
        lower, upper = fay_feuer_ci(
            asr_val, var_val, self.weights, self.person_years, alpha
        )
        self.assertLess(lower, asr_val)
        self.assertLess(asr_val, upper)
        self.assertAlmostEqual(lower, 0.004068, places=5)
        self.assertAlmostEqual(upper, 0.006081, places=5)

    def test_ci_zero_events_gives_zero_lower_bound(self):
        weights = [0.5, 0.5]
        counts = [0.0, 0.0]
        person_years = [1000.0, 1000.0]
        asr_val, var_val = direct_standardize(weights, counts, person_years)
        self.assertEqual(asr_val, 0.0)
        lower, upper = fay_feuer_ci(asr_val, var_val, weights, person_years)
        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, 0.0)

    def test_empty_or_invalid_weights_raises(self):
        with self.assertRaises(ValueError):
            direct_standardize([0.0, 0.0], [10.0, 20.0], [1000.0, 1000.0])


class TestIndirectStandardizationSMR(unittest.TestCase):
    def test_smr_matches_hand_calculation(self):
        study = AgeSpecificData(
            age_groups=["0-4", "5-9"],
            counts=[15, 5],
            person_years=[2500, 2500],
        )
        ref_rates = {"0-4": 0.002, "5-9": 0.002}
        observed, expected, smr = indirect_standardize(study, ref_rates)
        self.assertEqual(observed, 20.0)
        self.assertAlmostEqual(expected, 10.0, places=10)
        self.assertAlmostEqual(smr, 2.0, places=10)

    def test_smr_ci_matches_exact_poisson(self):
        observed, expected = 20.0, 10.0
        alpha = 0.05
        lower, upper = smr_poisson_ci(observed, expected, alpha)
        self.assertLess(lower, 2.0)
        self.assertLess(2.0, upper)
        self.assertAlmostEqual(lower, 1.22165, places=4)
        self.assertAlmostEqual(upper, 3.08884, places=4)

    def test_smr_zero_observed(self):
        lower, upper = smr_poisson_ci(0.0, 10.0)
        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, 0.0)


class TestHighLevelASRCalculator(unittest.TestCase):
    def test_calculate_direct_asr(self):
        data = AgeSpecificData(
            age_groups=["0-4", "5-9"],
            counts=[10.0, 20.0],
            person_years=[10000.0, 10000.0],
        )
        custom_std = [("0-4", 500.0), ("5-9", 500.0)]
        res = ASRCalculator.calculate_direct_asr(data, standard_population=custom_std)
        self.assertEqual(res.total_events, 30.0)
        self.assertEqual(res.total_person_years, 20000.0)
        self.assertAlmostEqual(res.asr_per_100k, 150.0, places=1)
        self.assertLess(res.fay_feuer_ci_95[0], res.asr_per_100k)
        self.assertGreater(res.fay_feuer_ci_95[1], res.asr_per_100k)

    def test_calculate_smr(self):
        data = AgeSpecificData(
            age_groups=["0-4", "5-9"],
            counts=[25.0, 15.0],
            person_years=[5000.0, 5000.0],
        )
        ref_rates = {"0-4": 0.002, "5-9": 0.002}
        res = ASRCalculator.calculate_smr(data, ref_rates)
        self.assertEqual(res.observed_events, 40.0)
        self.assertEqual(res.expected_events, 20.0)
        self.assertAlmostEqual(res.smr, 2.0, places=2)
        self.assertTrue("Statistically significant excess risk" in res.interpretation)

    def test_compare_standardized_rates(self):
        data1 = AgeSpecificData(["0-4", "5-9"], [20.0, 20.0], [10000.0, 10000.0])
        data2 = AgeSpecificData(["0-4", "5-9"], [10.0, 10.0], [10000.0, 10000.0])
        custom_std = [("0-4", 500.0), ("5-9", 500.0)]
        r1 = ASRCalculator.calculate_direct_asr(data1, standard_population=custom_std)
        r2 = ASRCalculator.calculate_direct_asr(data2, standard_population=custom_std)

        comp = ASRCalculator.compare_standardized_rates(r1, r2, pop1_name="Pop A", pop2_name="Pop B")
        self.assertAlmostEqual(comp.rate_ratio, 2.0, places=2)
        self.assertAlmostEqual(comp.rate_difference_per_100k, 100.0, places=1)


class TestRegressionCases(unittest.TestCase):
    def test_cumulative_rate_excludes_75_79_band(self):
        ages = [age for age, _ in asr.WHO_WORLD_2000_2025]
        counts = [0.0] * len(ages)
        counts[ages.index("75-79")] = 100.0
        data = AgeSpecificData(ages, counts, [100000.0] * len(ages))
        res = ASRCalculator.calculate_direct_asr(data, standard_population="who2000")
        self.assertEqual(res.cumulative_rate_0_74_pct, 0.0)
        self.assertEqual(res.cumulative_risk_0_74_pct, 0.0)

    def test_zero_observed_smr_has_positive_upper_byar_bound(self):
        data = AgeSpecificData(
            age_groups=["0-4", "5-9"],
            counts=[0.0, 0.0],
            person_years=[5000.0, 5000.0],
        )
        res = ASRCalculator.calculate_smr(data, {"0-4": 0.001, "5-9": 0.001})
        self.assertEqual(res.byar_ci_95[0], 0.0)
        self.assertGreater(res.byar_ci_95[1], 0.0)


class TestCLIAndBatch(unittest.TestCase):
    def test_cli_demo(self):
        self.assertEqual(cli.main(["--demo"]), 0)

    def test_cli_direct_sample_csv(self):
        sample_path = ROOT_DIR / "sample.csv"
        self.assertEqual(cli.main(["direct", "--input", str(sample_path)]), 0)

    def test_cli_direct_json(self):
        import io
        sample_path = ROOT_DIR / "sample.csv"
        out = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = out
        try:
            res = cli.main(["direct", "--input", str(sample_path), "--json"])
            self.assertEqual(res, 0)
        finally:
            sys.stdout = old_stdout

        data = json.loads(out.getvalue())
        self.assertIn("asr_per_100k", data)
        self.assertIn("fay_feuer_ci_95", data)

    def test_cli_batch_csv(self):
        sample_path = ROOT_DIR / "sample.csv"
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "out_batch.csv")
            ret = cli.main(["batch", "--input", str(sample_path), "--output", out_file])
            self.assertEqual(ret, 0)
            self.assertTrue(os.path.exists(out_file))

    def test_cli_compare(self):
        pop1 = ROOT_DIR / "examples" / "population_a.csv"
        pop2 = ROOT_DIR / "examples" / "population_b.csv"
        self.assertEqual(cli.main(["compare", "--pop1", str(pop1), "--pop2", str(pop2)]), 0)


class TestHelperFunctions(unittest.TestCase):
    """Tests for standard_weights, align_to_standard, direct_standardization_report."""

    def test_standard_weights_basic(self):
        rows = [("0-4", 100.0), ("5-9", 200.0), ("10-14", 300.0)]
        ages, weights = standard_weights(rows)
        self.assertEqual(ages, ["0-4", "5-9", "10-14"])
        self.assertAlmostEqual(sum(weights), 1.0, places=10)
        self.assertAlmostEqual(weights[0], 100.0 / 600.0, places=10)
        self.assertAlmostEqual(weights[1], 200.0 / 600.0, places=10)
        self.assertAlmostEqual(weights[2], 300.0 / 600.0, places=10)

    def test_standard_weights_empty_raises(self):
        with self.assertRaises(ValueError):
            standard_weights([])

    def test_standard_weights_zero_sum_raises(self):
        with self.assertRaises(ValueError):
            standard_weights([("0-4", 0.0), ("5-9", 0.0)])

    def test_align_to_standard_basic(self):
        data = AgeSpecificData(
            age_groups=["0-4", "5-9", "10-14"],
            counts=[10.0, 20.0, 30.0],
            person_years=[1000.0, 2000.0, 3000.0],
        )
        std_ages = ["0-4", "10-14"]
        std_weights = [0.5, 0.5]
        w, c, py = align_to_standard(data, std_ages, std_weights)
        self.assertEqual(w, [0.5, 0.5])
        self.assertEqual(c, [10.0, 30.0])
        self.assertEqual(py, [1000.0, 3000.0])

    def test_align_to_standard_missing_age_raises(self):
        data = AgeSpecificData(
            age_groups=["0-4", "5-9"],
            counts=[10.0, 20.0],
            person_years=[1000.0, 2000.0],
        )
        with self.assertRaises(ValueError):
            align_to_standard(data, ["0-4", "99+"], [0.5, 0.5])

    def test_direct_standardization_report(self):
        data = AgeSpecificData(
            age_groups=["0-4", "5-9"],
            counts=[10.0, 20.0],
            person_years=[10000.0, 10000.0],
        )
        std_rows = [("0-4", 500.0), ("5-9", 500.0)]
        rep = direct_standardization_report(data, std_rows)
        self.assertIn("asr", rep)
        self.assertIn("asr_raw", rep)
        self.assertIn("asr_lower", rep)
        self.assertIn("asr_upper", rep)
        self.assertIn("crude_rate", rep)
        self.assertGreater(rep["asr"], 0)
        self.assertLess(rep["asr_lower"], rep["asr"])
        self.assertGreater(rep["asr_upper"], rep["asr"])


class TestInputValidation(unittest.TestCase):
    """Tests for alpha parameter validation and CSV security."""

    def test_invalid_alpha_direct_standardize_raises(self):
        data = AgeSpecificData(["0-4"], [10.0], [1000.0])
        with self.assertRaises(ValueError):
            ASRCalculator.calculate_direct_asr(data, alpha=0.0)

    def test_invalid_alpha_fay_feuer_raises(self):
        with self.assertRaises(ValueError):
            fay_feuer_ci(0.001, 1e-8, [0.5, 0.5], [1000.0, 1000.0], alpha=1.0)

    def test_invalid_alpha_smr_raises(self):
        data = AgeSpecificData(["0-4"], [10.0], [1000.0])
        with self.assertRaises(ValueError):
            ASRCalculator.calculate_smr(data, {"0-4": 0.001}, alpha=-0.1)

    def test_invalid_alpha_poisson_ci_raises(self):
        with self.assertRaises(ValueError):
            smr_poisson_ci(10.0, 5.0, alpha=0.0)

    def test_csv_wrong_extension_raises(self):
        with self.assertRaises(ValueError):
            read_age_specific_csv("data.txt")

    def test_csv_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            read_age_specific_csv("nonexistent_file.csv")

    def test_csv_negative_count_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            f.write("age_group,count,person_years\n0-4,-5,1000\n")
            tmp_path = f.name
        try:
            with self.assertRaises(ValueError):
                read_age_specific_csv(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_csv_invalid_numeric_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            f.write("age_group,count,person_years\n0-4,abc,1000\n")
            tmp_path = f.name
        try:
            with self.assertRaises(ValueError):
                read_age_specific_csv(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_mismatched_lengths_direct_standardize(self):
        with self.assertRaises(ValueError):
            direct_standardize([0.5, 0.5], [10.0], [1000.0, 2000.0])


if __name__ == "__main__":
    unittest.main()
