#!/usr/bin/env python3
"""
Age-Standardized Rate Calculator
=================================

Computes directly age-standardized incidence or mortality rates with
Fay & Feuer (1997) gamma-distribution confidence intervals, indirect
standardization (SMR), and standardized rate ratios between two
populations.

Method
------
Direct standardization:
    ASR = sum_i( w_i * r_i )
where r_i = d_i / n_i is the age-specific rate (events d_i over
person-years n_i in age band i) and w_i is the proportion of the
standard population in age band i (sum_i w_i = 1).

Variance (Fay & Feuer 1997):
    Var(ASR) = sum_i( w_i^2 * d_i / n_i^2 )

Gamma-distribution confidence interval (Fay & Feuer 1997; the method
used by NCI SEER*Stat for age-standardized rates):
    wm = max_i( w_i / n_i )
    L  = (Var / (2*ASR)) * chi2.ppf(alpha/2,     df=2*ASR^2/Var)
    U  = ((Var+wm^2) / (2*(ASR+wm))) * chi2.ppf(1-alpha/2, df=2*(ASR+wm)^2/(Var+wm^2))

Indirect standardization / SMR:
    E   = sum_i( ref_rate_i * n_i )   (expected events under reference rates)
    SMR = O / E                        (O = total observed events)
CI on O is the exact Poisson (chi-square) interval, divided by E.

Standardized rate ratio (comparison mode):
    ratio = ASR1 / ASR2
CI via the log-transform delta method, using each rate's Fay-Feuer
variance:
    SE(log ratio) = sqrt(Var1/ASR1^2 + Var2/ASR2^2)
    CI = ratio * exp(+/- z_{1-alpha/2} * SE)
"""

import argparse
import csv
import sys

import numpy as np
from scipy.stats import chi2, norm

# ---------------------------------------------------------------------------
# Built-in standard populations (age_group -> standard population count)
# ---------------------------------------------------------------------------

# WHO World Standard Population 2000-2025 (Ahmad et al. 2001, WHO).
WHO_WORLD_2000_2025 = [
    ("0-4", 8860), ("5-9", 8690), ("10-14", 8600), ("15-19", 8470),
    ("20-24", 8220), ("25-29", 7930), ("30-34", 7610), ("35-39", 7150),
    ("40-44", 6590), ("45-49", 6040), ("50-54", 5370), ("55-59", 4550),
    ("60-64", 3720), ("65-69", 2960), ("70-74", 2210), ("75-79", 1520),
    ("80-84", 910), ("85+", 635),
]

# US 2000 Standard Population, 18 age groups (0-4 combined), NCI/SEER.
US_2000_STANDARD = [
    ("0-4", 18987000), ("5-9", 19920000), ("10-14", 20202000),
    ("15-19", 20092000), ("20-24", 19928000), ("25-29", 19703000),
    ("30-34", 20437000), ("35-39", 22842000), ("40-44", 22933000),
    ("45-49", 19983000), ("50-54", 17562000), ("55-59", 13434000),
    ("60-64", 10733000), ("65-69", 9480000), ("70-74", 8857000),
    ("75-79", 7415000), ("80-84", 4945000), ("85+", 4259000),
]

BUILTIN_STANDARDS = {
    "who2000": WHO_WORLD_2000_2025,
    "us2000": US_2000_STANDARD,
}


# ---------------------------------------------------------------------------
# Data structures / IO
# ---------------------------------------------------------------------------

class AgeSpecificData:
    """Age-specific event counts and person-years, in CSV row order."""

    def __init__(self, age_groups, counts, person_years):
        self.age_groups = list(age_groups)
        self.counts = np.asarray(counts, dtype=float)
        self.person_years = np.asarray(person_years, dtype=float)

    def crude_rate(self):
        total_py = self.person_years.sum()
        if total_py <= 0:
            raise ValueError("Total person-years must be positive.")
        return self.counts.sum() / total_py


def read_age_specific_csv(path):
    """Read a CSV with columns: age_group,count,person_years."""
    age_groups, counts, person_years = [], [], []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"age_group", "count", "person_years"}
        if reader.fieldnames is None or not required.issubset(
            {c.strip() for c in reader.fieldnames}
        ):
            raise ValueError(
                f"{path}: expected columns {sorted(required)}, "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            age_groups.append(row["age_group"].strip())
            counts.append(float(row["count"]))
            person_years.append(float(row["person_years"]))
    if not age_groups:
        raise ValueError(f"{path}: no data rows found.")
    return AgeSpecificData(age_groups, counts, person_years)


def read_standard_population_csv(path):
    """Read a CSV with columns: age_group,population."""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"age_group", "population"}
        if reader.fieldnames is None or not required.issubset(
            {c.strip() for c in reader.fieldnames}
        ):
            raise ValueError(
                f"{path}: expected columns {sorted(required)}, "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            rows.append((row["age_group"].strip(), float(row["population"])))
    if not rows:
        raise ValueError(f"{path}: no data rows found.")
    return rows


def load_standard_population(name_or_path):
    """Return a list of (age_group, population) for a built-in name or CSV path."""
    if name_or_path in BUILTIN_STANDARDS:
        return BUILTIN_STANDARDS[name_or_path]
    return read_standard_population_csv(name_or_path)


def standard_weights(std_pop_rows):
    """Convert (age_group, population) rows into (age_groups, normalized weights)."""
    age_groups = [r[0] for r in std_pop_rows]
    populations = np.asarray([r[1] for r in std_pop_rows], dtype=float)
    total = populations.sum()
    if total <= 0:
        raise ValueError("Standard population total must be positive.")
    return age_groups, populations / total


def align_to_standard(data: AgeSpecificData, std_age_groups, std_weights):
    """
    Reorder/validate `data` so its age bands match the standard population's
    age bands exactly (by label), returning (weights, counts, person_years)
    arrays in standard-population order.
    """
    index = {ag: i for i, ag in enumerate(data.age_groups)}
    missing = [ag for ag in std_age_groups if ag not in index]
    extra = [ag for ag in data.age_groups if ag not in std_age_groups]
    if missing:
        raise ValueError(
            "Input data is missing age band(s) required by the standard "
            f"population: {missing}"
        )
    if extra:
        raise ValueError(
            f"Input data has age band(s) not present in the standard "
            f"population: {extra}"
        )
    order = [index[ag] for ag in std_age_groups]
    counts = data.counts[order]
    person_years = data.person_years[order]
    return np.asarray(std_weights, dtype=float), counts, person_years


# ---------------------------------------------------------------------------
# Direct standardization + Fay-Feuer (1997) gamma confidence interval
# ---------------------------------------------------------------------------

def direct_standardize(weights, counts, person_years):
    """
    Direct age-standardization.

    Returns (asr, variance) in raw per-person-year units, where
    asr = sum(w_i * d_i/n_i) and variance = sum(w_i^2 * d_i/n_i^2)
    (Fay & Feuer 1997).
    """
    if np.any(person_years <= 0):
        raise ValueError("All age-band person-years must be positive.")
    rates = counts / person_years
    asr = float(np.sum(weights * rates))
    variance = float(np.sum((weights ** 2) * counts / (person_years ** 2)))
    return asr, variance


def fay_feuer_ci(asr, variance, weights, person_years, alpha=0.05):
    """
    Fay & Feuer (1997) gamma-distribution confidence interval for a
    directly age-standardized rate. Returns (lower, upper) in the same
    raw units as `asr`/`variance`.
    """
    wm = float(np.max(weights / person_years))

    if asr <= 0 or variance <= 0:
        lower = 0.0
    else:
        df_l = 2.0 * asr ** 2 / variance
        lower = (variance / (2.0 * asr)) * chi2.ppf(alpha / 2.0, df_l)

    df_u = 2.0 * (asr + wm) ** 2 / (variance + wm ** 2)
    upper = ((variance + wm ** 2) / (2.0 * (asr + wm))) * chi2.ppf(
        1.0 - alpha / 2.0, df_u
    )
    return lower, upper


def direct_standardization_report(data, std_rows, alpha=0.05, per=100000.0):
    """
    Full direct-standardization result for one population.

    Returns a dict with age_groups, crude rate, standardized rate and CI,
    all scaled to `per` person-years for display.
    """
    std_age_groups, weights = standard_weights(std_rows)
    w, counts, person_years = align_to_standard(data, std_age_groups, weights)

    asr_raw, var_raw = direct_standardize(w, counts, person_years)
    lower_raw, upper_raw = fay_feuer_ci(asr_raw, var_raw, w, person_years, alpha)

    return {
        "age_groups": std_age_groups,
        "weights": w,
        "counts": counts,
        "person_years": person_years,
        "crude_rate": data.crude_rate() * per,
        "asr": asr_raw * per,
        "asr_lower": lower_raw * per,
        "asr_upper": upper_raw * per,
        "variance_raw": var_raw,
        "asr_raw": asr_raw,
        "alpha": alpha,
        "per": per,
    }


# ---------------------------------------------------------------------------
# Indirect standardization (SMR)
# ---------------------------------------------------------------------------

def indirect_standardize(study: AgeSpecificData, reference_rates_by_group):
    """
    Indirect standardization: apply reference age-specific rates to the
    study population's person-years to obtain expected events, then the
    standardized mortality/morbidity ratio SMR = observed / expected.

    `reference_rates_by_group` is a dict age_group -> rate (per person-year).
    Returns (observed, expected, smr).
    """
    missing = [ag for ag in study.age_groups if ag not in reference_rates_by_group]
    if missing:
        raise ValueError(
            f"Reference rates missing age band(s) present in study data: {missing}"
        )
    ref_rates = np.array(
        [reference_rates_by_group[ag] for ag in study.age_groups], dtype=float
    )
    expected = float(np.sum(ref_rates * study.person_years))
    observed = float(np.sum(study.counts))
    if expected <= 0:
        raise ValueError("Expected event count must be positive to compute an SMR.")
    smr = observed / expected
    return observed, expected, smr


def smr_poisson_ci(observed, expected, alpha=0.05):
    """
    Exact Poisson (chi-square-based) confidence interval for the SMR,
    obtained by dividing the exact Poisson CI for the observed count by
    the expected count.
    """
    if observed == 0:
        lower = 0.0
    else:
        lower = chi2.ppf(alpha / 2.0, 2 * observed) / 2.0
    upper = chi2.ppf(1.0 - alpha / 2.0, 2 * (observed + 1)) / 2.0
    return lower / expected, upper / expected


def reference_rates_from_counts(ref_data: AgeSpecificData):
    """Build an age_group -> rate dict from a reference population's counts/PY."""
    if np.any(ref_data.person_years <= 0):
        raise ValueError("Reference person-years must be positive.")
    rates = ref_data.counts / ref_data.person_years
    return dict(zip(ref_data.age_groups, rates))


def read_reference_rates_csv(path):
    """Read a CSV with columns: age_group,rate (rate per person-year)."""
    rates = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"age_group", "rate"}
        if reader.fieldnames is None or not required.issubset(
            {c.strip() for c in reader.fieldnames}
        ):
            raise ValueError(
                f"{path}: expected columns {sorted(required)}, "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            rates[row["age_group"].strip()] = float(row["rate"])
    if not rates:
        raise ValueError(f"{path}: no data rows found.")
    return rates


# ---------------------------------------------------------------------------
# Standardized rate ratio (comparison mode)
# ---------------------------------------------------------------------------

def rate_ratio_ci(asr1_raw, var1_raw, asr2_raw, var2_raw, alpha=0.05):
    """
    Standardized rate ratio ASR1/ASR2 with a log-transform delta-method
    confidence interval. Works in raw (unscaled) units; the ratio is
    scale-invariant as long as both rates use the same "per" denominator.
    """
    if asr1_raw <= 0 or asr2_raw <= 0:
        raise ValueError("Both standardized rates must be positive to form a ratio.")
    ratio = asr1_raw / asr2_raw
    se_log = np.sqrt(var1_raw / asr1_raw ** 2 + var2_raw / asr2_raw ** 2)
    z = norm.ppf(1.0 - alpha / 2.0)
    lower = ratio * np.exp(-z * se_log)
    upper = ratio * np.exp(z * se_log)
    return float(ratio), float(lower), float(upper)


# ---------------------------------------------------------------------------
# Reporting: summary table + bar chart
# ---------------------------------------------------------------------------

def print_direct_summary(label, result):
    per = result["per"]
    print(f"\n=== Direct age-standardization: {label} ===")
    print(f"Standard population age bands: {len(result['age_groups'])}")
    print(f"{'Age group':<12}{'Count':>10}{'Person-yrs':>14}{'Crude rate/PY':>16}")
    for ag, c, py in zip(
        result["age_groups"], result["counts"], result["person_years"]
    ):
        rate = c / py if py > 0 else float("nan")
        print(f"{ag:<12}{c:>10.0f}{py:>14.0f}{rate:>16.6f}")
    ci_pct = int(round((1 - result["alpha"]) * 100))
    print(f"\nCrude rate:               {result['crude_rate']:.3f} per {per:,.0f} person-years")
    print(
        f"Age-standardized rate:    {result['asr']:.3f} per {per:,.0f} person-years "
        f"({ci_pct}% CI: {result['asr_lower']:.3f} - {result['asr_upper']:.3f})"
    )


def plot_crude_vs_standardized(label, result, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    values = [result["crude_rate"], result["asr"]]
    errors = [[0, result["asr"] - result["asr_lower"]],
              [0, result["asr_upper"] - result["asr"]]]
    bars = ax.bar(
        ["Crude rate", "Standardized rate"],
        values,
        yerr=errors,
        capsize=6,
        color=["#7f8fa6", "#2d5f8b"],
    )
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel(f"Rate per {result['per']:,.0f} person-years")
    ax.set_title(f"Crude vs. age-standardized rate\n{label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_direct(args):
    data = read_age_specific_csv(args.input)
    std_rows = load_standard_population(args.standard)
    result = direct_standardization_report(data, std_rows, alpha=args.alpha, per=args.per)
    label = args.label or args.input
    print_direct_summary(label, result)
    if args.output:
        plot_crude_vs_standardized(label, result, args.output)
        print(f"\nChart saved to: {args.output}")


def cmd_indirect(args):
    study = read_age_specific_csv(args.input)

    if args.reference_rates:
        ref_rates = read_reference_rates_csv(args.reference_rates)
    elif args.reference_input:
        ref_data = read_age_specific_csv(args.reference_input)
        ref_rates = reference_rates_from_counts(ref_data)
    else:
        raise ValueError("Provide --reference-rates or --reference-input.")

    observed, expected, smr = indirect_standardize(study, ref_rates)
    lower, upper = smr_poisson_ci(observed, expected, args.alpha)

    ci_pct = int(round((1 - args.alpha) * 100))
    label = args.label or args.input
    print(f"\n=== Indirect standardization (SMR): {label} ===")
    print(f"Observed events:  {observed:.0f}")
    print(f"Expected events:  {expected:.4f}")
    print(f"SMR:              {smr:.4f} ({ci_pct}% CI: {lower:.4f} - {upper:.4f})")
    if smr > 1 and lower > 1:
        print("Interpretation: significantly elevated risk vs. reference population.")
    elif smr < 1 and upper < 1:
        print("Interpretation: significantly reduced risk vs. reference population.")
    else:
        print("Interpretation: not significantly different from the reference population.")


def cmd_compare(args):
    data1 = read_age_specific_csv(args.input1)
    data2 = read_age_specific_csv(args.input2)
    std_rows = load_standard_population(args.standard)

    result1 = direct_standardization_report(data1, std_rows, alpha=args.alpha, per=args.per)
    result2 = direct_standardization_report(data2, std_rows, alpha=args.alpha, per=args.per)

    label1 = args.label1 or args.input1
    label2 = args.label2 or args.input2
    print_direct_summary(label1, result1)
    print_direct_summary(label2, result2)

    ratio, lower, upper = rate_ratio_ci(
        result1["asr_raw"], result1["variance_raw"],
        result2["asr_raw"], result2["variance_raw"],
        alpha=args.alpha,
    )
    ci_pct = int(round((1 - args.alpha) * 100))
    print(f"\n=== Standardized rate ratio: {label1} / {label2} ===")
    print(f"Rate ratio: {ratio:.4f} ({ci_pct}% CI: {lower:.4f} - {upper:.4f})")
    if lower > 1:
        print(f"Interpretation: {label1} rate is significantly higher than {label2}.")
    elif upper < 1:
        print(f"Interpretation: {label1} rate is significantly lower than {label2}.")
    else:
        print("Interpretation: no statistically significant difference in rates.")

    if args.output:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        groups = ["Crude", "Standardized"]
        x = np.arange(len(groups))
        width = 0.35
        vals1 = [result1["crude_rate"], result1["asr"]]
        vals2 = [result2["crude_rate"], result2["asr"]]
        err1 = [[0, result1["asr"] - result1["asr_lower"]],
                [0, result1["asr_upper"] - result1["asr"]]]
        err2 = [[0, result2["asr"] - result2["asr_lower"]],
                [0, result2["asr_upper"] - result2["asr"]]]
        ax.bar(x - width / 2, vals1, width, yerr=err1, capsize=5, label=label1, color="#2d5f8b")
        ax.bar(x + width / 2, vals2, width, yerr=err2, capsize=5, label=label2, color="#c46a2f")
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel(f"Rate per {args.per:,.0f} person-years")
        ax.set_title("Crude vs. age-standardized rate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.output, dpi=150)
        plt.close(fig)
        print(f"\nChart saved to: {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        prog="asr_calculator",
        description="Age-Standardized Rate Calculator: direct/indirect age "
        "standardization with Fay-Feuer (1997) confidence intervals.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_direct = sub.add_parser(
        "direct", help="Directly age-standardize a population's rate."
    )
    p_direct.add_argument("--input", required=True, help="CSV: age_group,count,person_years")
    p_direct.add_argument(
        "--standard", default="who2000",
        help="Built-in standard ('who2000', 'us2000') or path to a custom "
        "CSV with columns age_group,population. Default: who2000",
    )
    p_direct.add_argument("--alpha", type=float, default=0.05, help="Significance level (default 0.05 -> 95%% CI)")
    p_direct.add_argument("--per", type=float, default=100000.0, help="Report rates per N person-years (default 100000)")
    p_direct.add_argument("--label", default=None, help="Label for the population in output")
    p_direct.add_argument("--output", default=None, help="Path to save crude-vs-standardized bar chart (PNG)")
    p_direct.set_defaults(func=cmd_direct)

    p_indirect = sub.add_parser(
        "indirect", help="Indirect standardization: compute an SMR."
    )
    p_indirect.add_argument("--input", required=True, help="Study population CSV: age_group,count,person_years")
    p_indirect.add_argument("--reference-rates", default=None, help="Reference rates CSV: age_group,rate")
    p_indirect.add_argument("--reference-input", default=None, help="Reference population CSV: age_group,count,person_years")
    p_indirect.add_argument("--alpha", type=float, default=0.05, help="Significance level (default 0.05 -> 95%% CI)")
    p_indirect.add_argument("--label", default=None, help="Label for the study population in output")
    p_indirect.set_defaults(func=cmd_indirect)

    p_compare = sub.add_parser(
        "compare", help="Standardized rate ratio between two populations."
    )
    p_compare.add_argument("--input1", required=True, help="Population 1 CSV: age_group,count,person_years")
    p_compare.add_argument("--input2", required=True, help="Population 2 CSV: age_group,count,person_years")
    p_compare.add_argument(
        "--standard", default="who2000",
        help="Built-in standard ('who2000', 'us2000') or path to a custom "
        "CSV with columns age_group,population. Default: who2000",
    )
    p_compare.add_argument("--alpha", type=float, default=0.05, help="Significance level (default 0.05 -> 95%% CI)")
    p_compare.add_argument("--per", type=float, default=100000.0, help="Report rates per N person-years (default 100000)")
    p_compare.add_argument("--label1", default=None, help="Label for population 1")
    p_compare.add_argument("--label2", default=None, help="Label for population 2")
    p_compare.add_argument("--output", default=None, help="Path to save comparison bar chart (PNG)")
    p_compare.set_defaults(func=cmd_compare)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
