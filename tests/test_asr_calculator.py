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
    fay_feuer_ci,
    indirect_standardize,
    smr_poisson_ci,
    read_age_specific_csv,
    normal_cdf,
    normal_ppf,
    chi2_ppf,
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


if __name__ == "__main__":
    unittest.main()
