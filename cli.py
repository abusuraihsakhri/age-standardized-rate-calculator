#!/usr/bin/env python3
"""
Command-Line Interface for Age-Standardized Rate (ASR) Calculator
=================================================================
Provides interactive and scriptable workflows for directly and indirectly
standardized rates, Fay & Feuer (1997) confidence intervals, and population comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from typing import List, Optional

from asr_calculator import (
    AgeSpecificData,
    ASRCalculator,
    BUILTIN_STANDARDS,
    WHO_WORLD_2000_2025,
    read_age_specific_csv,
    DirectStandardizationResult,
    IndirectStandardizationResult,
    RateRatioComparisonResult,
)


def format_direct_report(res: DirectStandardizationResult) -> str:
    lines = [
        "=" * 78,
        f" AGE-STANDARDIZED RATE (ASR) REPORT - STANDARD: {res.standard_name.upper()}",
        "=" * 78,
        f"Summary Overview:",
        f"  - Total Observed Events:          {res.total_events:,.0f}",
        f"  - Total Population Person-Years:  {res.total_person_years:,.0f}",
        f"  - Crude Rate:                     {res.crude_rate_per_100k:.2f} per 100,000",
        f"  - Age-Standardized Rate (ASR):    {res.asr_per_100k:.2f} per 100,000",
        f"  - Standard Error (SE):            {res.standard_error_per_100k:.4f} per 100,000",
        "-" * 78,
        "Confidence Intervals (95% CI):",
        f"  - Fay & Feuer (1997) Gamma CI:    [{res.fay_feuer_ci_95[0]:.2f} - {res.fay_feuer_ci_95[1]:.2f}] per 100k (SEER Gold Standard)",
        f"  - Normal Approximation (Wald):    [{res.wald_ci_95[0]:.2f} - {res.wald_ci_95[1]:.2f}] per 100k",
        f"  - Log-Transformed CI:             [{res.log_transformed_ci_95[0]:.2f} - {res.log_transformed_ci_95[1]:.2f}] per 100k",
        "-" * 78,
        "Cumulative Risk Metrics (0 to 74 Years):",
        f"  - Cumulative Rate (0-74):         {res.cumulative_rate_0_74_pct:.2f}%",
        f"  - Cumulative Lifetime Risk (0-74):{res.cumulative_risk_0_74_pct:.2f}%",
        "=" * 78,
    ]
    return "\n".join(lines)


def format_indirect_report(res: IndirectStandardizationResult) -> str:
    lines = [
        "=" * 78,
        " INDIRECT STANDARDIZATION & STANDARDIZED MORTALITY/INCIDENCE RATIO (SMR/SIR)",
        "=" * 78,
        f"  - Total Observed Events (O):      {res.observed_events:,.0f}",
        f"  - Total Expected Events (E):      {res.expected_events:,.2f}",
        f"  - Standardized Ratio (SMR/SIR):   {res.smr:.3f}",
        f"  - Exact Poisson 95% CI:           [{res.exact_poisson_ci_95[0]:.3f} - {res.exact_poisson_ci_95[1]:.3f}]",
        f"  - Byar Approximation 95% CI:      [{res.byar_ci_95[0]:.3f} - {res.byar_ci_95[1]:.3f}]",
        f"  - Two-Sided p-Value (vs. 1.0):    {res.p_value_vs_unity:.4f}",
        f"  - Clinical Interpretation:        {res.interpretation}",
        "=" * 78,
    ]
    return "\n".join(lines)


def format_compare_report(res: RateRatioComparisonResult) -> str:
    lines = [
        "=" * 78,
        f" STANDARDIZED POPULATION COMPARISON: {res.population_1_name} vs. {res.population_2_name}",
        "=" * 78,
        f"  - ASR ({res.population_1_name}):      {res.asr_1_per_100k:.2f} per 100,000",
        f"  - ASR ({res.population_2_name}):      {res.asr_2_per_100k:.2f} per 100,000",
        f"  - Standardized Rate Ratio (SRR):  {res.rate_ratio:.3f} [95% CI: {res.rate_ratio_ci_95[0]:.3f} - {res.rate_ratio_ci_95[1]:.3f}]",
        f"  - Standardized Rate Difference:   {res.rate_difference_per_100k:+.2f} per 100k [95% CI: {res.rate_difference_ci_95[0]:+.2f} to {res.rate_difference_ci_95[1]:+.2f}]",
        f"  - Two-Sided p-Value:              {res.two_sided_p_value:.4f}",
        f"  - Statistically Significant:      {'YES (p < 0.05)' if res.statistically_significant else 'NO'}",
        "=" * 78,
    ]
    return "\n".join(lines)


def run_demo(as_json: bool = False) -> int:
    """Run benchmark demonstration comparing population A vs B."""
    # Synthetic realistic standard age-stratified data
    age_groups = [g for g, _ in WHO_WORLD_2000_2025]
    # Population A (Higher cancer rate in older ages)
    py_a = [100000.0] * len(age_groups)
    counts_a = [5, 3, 2, 4, 8, 12, 25, 45, 80, 140, 220, 350, 520, 710, 890, 1050, 1120, 1200]
    data_a = AgeSpecificData(age_groups, [float(c) for c in counts_a], py_a)

    # Population B (Lower cancer rate)
    py_b = [100000.0] * len(age_groups)
    counts_b = [4, 2, 2, 3, 6, 9, 18, 32, 58, 95, 150, 240, 360, 500, 620, 740, 810, 880]
    data_b = AgeSpecificData(age_groups, [float(c) for c in counts_b], py_b)

    res_a_who = ASRCalculator.calculate_direct_asr(data_a, standard_population="who2000")
    res_b_who = ASRCalculator.calculate_direct_asr(data_b, standard_population="who2000")
    res_a_us = ASRCalculator.calculate_direct_asr(data_a, standard_population="us2000")

    comp = ASRCalculator.compare_standardized_rates(res_a_who, res_b_who, "Region North", "Region South")

    # SMR test
    ref_rates = {age: (cnt / py) for age, cnt, py in zip(age_groups, counts_b, py_b)}
    smr_res = ASRCalculator.calculate_smr(data_a, ref_rates)

    if as_json:
        out = {
            "pop_a_who": asdict(res_a_who),
            "pop_b_who": asdict(res_b_who),
            "pop_a_us2000": asdict(res_a_us),
            "comparison": asdict(comp),
            "smr": asdict(smr_res),
        }
        print(json.dumps(out, indent=2))
        return 0

    print("=" * 80)
    print(" AGE-STANDARDIZED RATE (ASR) CALCULATOR - BENCHMARK SUITE")
    print("=" * 80)
    print(format_direct_report(res_a_who))
    print()
    print(format_direct_report(res_a_us))
    print()
    print(format_compare_report(comp))
    print()
    print(format_indirect_report(smr_res))
    return 0


def run_interactive() -> int:
    """Interactive studio."""
    print("=" * 70)
    print(" Age-Standardized Rate (ASR) Studio")
    print("=" * 70)
    print("1. Run Complete Benchmark Demonstration (Direct ASR, US/WHO, SMR, Comparison)")
    print("2. Choose Standard Population to View Weights")
    print("q. Exit")
    print("-" * 70)

    choice = input("Select an option [1-2, q]: ").strip()
    if choice in ("q", "quit", "exit"):
        return 0

    if choice == "1":
        run_demo()
    elif choice == "2":
        for k in BUILTIN_STANDARDS:
            print(f"  * {k}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asr_calculator",
        description="Age-Standardized Rate (ASR) & SMR Calculator",
    )
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive studio")
    parser.add_argument("--demo", action="store_true", help="Run reference comparison benchmark")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    sub = parser.add_subparsers(dest="command", help="Subcommands")

    # Direct ASR
    dir_p = sub.add_parser("direct", help="Direct age-standardization from CSV")
    dir_p.add_argument("--input", "-in", required=True, help="Input CSV path (age_group, count, person_years)")
    dir_p.add_argument("--standard", "-std", choices=list(BUILTIN_STANDARDS.keys()), default="who2000", help="Standard population")
    dir_p.add_argument("--alpha", type=float, default=0.05, help="Significance level (default 0.05 for 95% CI)")
    dir_p.add_argument("--json", action="store_true", help="Output results in JSON format")

    # Compare
    cmp_p = sub.add_parser("compare", help="Compare standardized rates between two CSV populations")
    cmp_p.add_argument("--pop1", required=True, help="Population 1 CSV path")
    cmp_p.add_argument("--pop2", required=True, help="Population 2 CSV path")
    cmp_p.add_argument("--name1", default="Population 1", help="Label for Population 1")
    cmp_p.add_argument("--name2", default="Population 2", help="Label for Population 2")
    cmp_p.add_argument("--standard", choices=list(BUILTIN_STANDARDS.keys()), default="who2000")
    cmp_p.add_argument("--json", action="store_true", help="Output results in JSON format")

    # Batch output
    b_p = sub.add_parser("batch", help="Batch compute direct ASR and export results CSV")
    b_p.add_argument("--input", "-in", required=True, help="Input CSV path")
    b_p.add_argument("--output", "-out", required=True, help="Output CSV path")
    b_p.add_argument("--standard", choices=list(BUILTIN_STANDARDS.keys()), default="who2000")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.interactive or (not args.command and not args.demo):
        if not args.demo and (argv is None or len(argv) == 0):
            return run_interactive()

    if args.demo:
        return run_demo(as_json=args.json)

    if args.command == "direct":
        try:
            data = read_age_specific_csv(args.input)
            res = ASRCalculator.calculate_direct_asr(data, standard_population=args.standard, alpha=args.alpha)
            if args.json:
                print(json.dumps(asdict(res), indent=2))
            else:
                print(format_direct_report(res))
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    if args.command == "compare":
        try:
            d1 = read_age_specific_csv(args.pop1)
            d2 = read_age_specific_csv(args.pop2)
            r1 = ASRCalculator.calculate_direct_asr(d1, standard_population=args.standard)
            r2 = ASRCalculator.calculate_direct_asr(d2, standard_population=args.standard)
            cmp_res = ASRCalculator.compare_standardized_rates(r1, r2, args.name1, args.name2)
            if args.json:
                print(json.dumps(asdict(cmp_res), indent=2))
            else:
                print(format_compare_report(cmp_res))
            return 0
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    if args.command == "batch":
        try:
            data = read_age_specific_csv(args.input)
            res = ASRCalculator.calculate_direct_asr(data, standard_population=args.standard)
            with open(args.output, "w", newline="", encoding="utf-8") as f_out:
                writer = csv.DictWriter(f_out, fieldnames=[
                    "standard", "crude_rate_per_100k", "asr_per_100k", "standard_error_per_100k",
                    "fay_feuer_ci_lower", "fay_feuer_ci_upper", "wald_ci_lower", "wald_ci_upper",
                    "cum_rate_0_74_pct", "cum_risk_0_74_pct"
                ])
                writer.writeheader()
                writer.writerow({
                    "standard": res.standard_name,
                    "crude_rate_per_100k": res.crude_rate_per_100k,
                    "asr_per_100k": res.asr_per_100k,
                    "standard_error_per_100k": res.standard_error_per_100k,
                    "fay_feuer_ci_lower": res.fay_feuer_ci_95[0],
                    "fay_feuer_ci_upper": res.fay_feuer_ci_95[1],
                    "wald_ci_lower": res.wald_ci_95[0],
                    "wald_ci_upper": res.wald_ci_95[1],
                    "cum_rate_0_74_pct": res.cumulative_rate_0_74_pct,
                    "cum_risk_0_74_pct": res.cumulative_risk_0_74_pct,
                })
            print(f"Successfully wrote ASR results to {args.output}")
            return 0
        except Exception as e:
            print(f"Batch error: {e}", file=sys.stderr)
            return 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
