#!/usr/bin/env python3
"""
Enrichment Features for Age-Standardized Rate Calculator
========================================================

Implements three enrichment features from specifications:

1. Multi-Standard Population Comparison
   Compute directly age-standardized rates for the same population under
   WHO World 2000-2025, US 2000, and European Standard Population 2013
   side by side, showing how standard choice changes the summary rate.

2. Time-Series Age Standardization with Trend Analysis
   Annual ASR estimation across calendar years with Estimated Annual
   Percent Change (EAPC) via log-linear regression and a simple grid-search
   joinpoint detector for trend inflection.

3. Bootstrap Confidence Intervals for Standardized Rate Ratios
   Nonparametric resampling of age-band counts to propagate count
   uncertainty through the ratio of two ASRs, complementing the
   delta-method interval in asr_calculator.py.
"""

import math

import numpy as np

try:
    from asr_calculator import (
        AgeSpecificData,
        BUILTIN_STANDARDS,
        EUROPEAN_2013_STANDARD,
        direct_standardization_report,
    )
except ImportError:  # allow running from repo root
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from asr_calculator import (
        AgeSpecificData,
        BUILTIN_STANDARDS,
        EUROPEAN_2013_STANDARD,
        direct_standardization_report,
    )


# ---------------------------------------------------------------------------
# 1. Multi-standard population comparison
# ---------------------------------------------------------------------------

# European Standard Population 2013 (ESP 2013), 18 five-year bands.
# Source: EUROPEAN_2013_STANDARD in asr_calculator.py (authoritative definition).
EUROPEAN_STANDARD_2013 = EUROPEAN_2013_STANDARD


def all_standard_populations():
    """Built-in standards plus ESP2013 keyed for comparison mode."""
    standards = dict(BUILTIN_STANDARDS)
    standards["esp2013"] = EUROPEAN_STANDARD_2013
    return standards


def multi_standard_comparison(data, alpha=0.05, per=100000.0):
    """
    Standardize one population against every built-in standard.

    Returns a list of dicts, one per standard, each carrying the ASR and
    Fay-Feuer CI plus the ratio of each ASR to the WHO-world baseline so
    users can see the effect of standard-population age structure alone.
    """
    if not isinstance(data, AgeSpecificData):
        raise TypeError("data must be an AgeSpecificData instance")
    results = []
    for name, rows in sorted(all_standard_populations().items()):
        try:
            rep = direct_standardization_report(data, rows, alpha=alpha, per=per)
        except ValueError:
            # Standards whose age bands do not exactly match the data are skipped
            continue
        results.append(
            {
                "standard": name,
                "asr": rep["asr"],
                "lower": rep["asr_lower"],
                "upper": rep["asr_upper"],
                "crude_rate": rep["crude_rate"],
            }
        )
    who = next((r for r in results if r["standard"] == "who2000"), None)
    if who is not None and who["asr"] > 0:
        for r in results:
            r["ratio_to_who"] = r["asr"] / who["asr"]
    return results


# ---------------------------------------------------------------------------
# 2. Time-series standardization with EAPC and joinpoint detection
# ---------------------------------------------------------------------------

def annual_asr_series(yearly_data, std_rows, alpha=0.05, per=100000.0):
    """
    Standardize a sequence of annual populations against one standard.

    `yearly_data` maps year -> AgeSpecificData. Returns a list of dicts
    ordered by year with the ASR and CI for each year.
    """
    series = []
    for year in sorted(yearly_data.keys()):
        rep = direct_standardization_report(
            yearly_data[year], std_rows, alpha=alpha, per=per
        )
        series.append(
            {
                "year": int(year),
                "asr": rep["asr"],
                "lower": rep["asr_lower"],
                "upper": rep["asr_upper"],
            }
        )
    return series


def eapc(series):
    """
    Estimated Annual Percent Change over a full ASR series, fitted by
    ordinary least squares on log(ASR):

        log(ASR_t) = a + b * t      =>   EAPC = 100 * (exp(b) - 1)

    Returns (eapc_percent, ci_lower, ci_upper) where the CI comes from the
    regression slope's standard error.
    """
    if len(series) < 3:
        raise ValueError("Need at least 3 years of data for an EAPC estimate.")
    years = np.array([s["year"] for s in series], dtype=float)
    rates = np.array([s["asr"] for s in series], dtype=float)
    mask = rates > 0
    if mask.sum() < 3:
        raise ValueError("At least 3 years must have positive rates.")
    x = years[mask] - years[mask].mean()
    y = np.log(rates[mask])
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = intercept + slope * x
    dof = len(x) - 2
    if dof < 1:
        raise ValueError("Not enough observations for slope inference.")
    resid_ss = float(np.sum((y - y_hat) ** 2))
    se_slope = math.sqrt(resid_ss / dof / float(np.sum((x - x.mean()) ** 2)))
    z = 1.959963984540054  # two-sided 95%
    eapc = 100.0 * (math.exp(slope) - 1.0)
    lo = 100.0 * (math.exp(slope - z * se_slope) - 1.0)
    hi = 100.0 * (math.exp(slope + z * se_slope) - 1.0)
    return eapc, lo, hi


def joinpoint_scan(series, min_segment=4):
    """
    Grid-search joinpoint detection: split the series at every candidate
    year and keep the split minimizing total residual sum of squares from
    two independent log-linear fits, provided it improves on a single fit
    by more than a 10% RSS reduction (a conservative significance proxy).

    Returns (best_year_or_None, improvement_fraction).
    """
    n = len(series)
    if n < 2 * min_segment:
        return None, 0.0
    years = np.array([s["year"] for s in series], dtype=float)
    logs = np.log(np.maximum([s["asr"] for s in series], 1e-12))

    def _fit_rss(xs, ys):
        if len(xs) < 3:
            return None
        b, a = np.polyfit(xs - xs.mean(), ys, 1)
        return float(np.sum((ys - (a + b * (xs - xs.mean()))) ** 2))

    rss_full = _fit_rss(years, logs)
    best_year, best_rss = None, rss_full
    for k in range(min_segment, n - min_segment + 1):
        r1 = _fit_rss(years[:k], logs[:k])
        r2 = _fit_rss(years[k:], logs[k:])
        if r1 is None or r2 is None or rss_full is None or rss_full <= 0:
            continue
        total = r1 + r2
        if total < best_rss:
            best_rss, best_year = total, int(years[k])
    if best_year is None or rss_full <= 0:
        return None, 0.0
    improvement = 1.0 - best_rss / rss_full
    if improvement < 0.10:
        return None, improvement
    return best_year, improvement


# ---------------------------------------------------------------------------
# 3. Bootstrap confidence interval for a standardized rate ratio
# ---------------------------------------------------------------------------

def bootstrap_rate_ratio_ci(
    data1, data2, std_rows, n_boot=2000, alpha=0.05, seed=42, per=100000.0
):
    """
    Percentile-bootstrap CI for ASR1/ASR2 by Poisson-resampling each
    age-band count around its observed value (preserving the mean while
    introducing count-level uncertainty), recomputing both ASRs on every
    replicate.

    Returns (ratio_point_estimate, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)

    def _aligned_arrays(data):
        from asr_calculator import standard_weights, align_to_standard

        _, w = standard_weights(std_rows)
        w, c, py = align_to_standard(data, [r[0] for r in std_rows], w)
        return np.asarray(w), np.asarray(c), np.asarray(py)

    w1, c1, py1 = _aligned_arrays(data1)
    w2, c2, py2 = _aligned_arrays(data2)
    point = direct_standardization_report(
        data1, std_rows, per=per
    )["asr_raw"] / direct_standardization_report(data2, std_rows, per=per)[
        "asr_raw"
    ]

    ratios = np.empty(n_boot)
    for i in range(n_boot):
        s1 = rng.poisson(c1).astype(float)
        s2 = rng.poisson(c2).astype(float)
        a1 = float(np.sum(w1 * s1 / py1))
        a2 = float(np.sum(w2 * s2 / py2))
        ratios[i] = a1 / a2 if a2 > 0 else np.nan
    valid = ratios[~np.isnan(ratios)]
    if len(valid) == 0:
        raise ValueError("All bootstrap replicates failed; check input data.")
    lower = float(np.quantile(valid, alpha / 2.0))
    upper = float(np.quantile(valid, 1.0 - alpha / 2.0))
    return float(point), lower, upper


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo_population(base_rate=0.004, scale=1.0, seed=7):
    """Synthetic 18-band population with plausible counts/person-years."""
    rng = np.random.default_rng(seed)
    bands = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34",
             "35-39", "40-44", "45-49", "50-54", "55-59", "60-64",
             "65-69", "70-74", "75-79", "80-84", "85+"]
    band_py = np.array([2.0, 2.1, 2.2, 2.1, 2.0, 1.9, 1.8, 1.6, 1.5,
                        1.4, 1.2, 1.0, 0.9, 0.7, 0.5, 0.35, 0.22, 0.12]) * 1e6
    # Rate rises steeply with age (cancer-like profile)
    age_factor = np.linspace(0.15, 8.0, len(bands))
    rates = base_rate * age_factor * scale
    counts = rng.poisson(rates * band_py)
    return AgeSpecificData(bands, counts, band_py)


if __name__ == "__main__":
    print("=" * 64)
    print("Feature 1: multi-standard comparison")
    print("=" * 64)
    pop = _demo_population()
    for row in multi_standard_comparison(pop):
        ratio = f"  x{row['ratio_to_who']:.3f} vs WHO" if "ratio_to_who" in row else ""
        print(
            f"{row['standard']:>8}: ASR {row['asr']:8.2f} "
            f"(CI {row['lower']:7.2f}-{row['upper']:7.2f}){ratio}"
        )

    print()
    print("=" * 64)
    print("Feature 2: time-series ASR with EAPC + joinpoint scan")
    print("=" * 64)
    yearly = {}
    for i, yr in enumerate(range(2010, 2020)):
        # Rising epidemic until 2016 then declining (joinpoint at 2016)
        scale = 1.0 + 0.06 * i if yr <= 2016 else 1.36 - 0.04 * (i - 6)
        yearly[yr] = _demo_population(scale=scale, seed=100 + i)
    std_rows = all_standard_populations()["who2000"]
    series = annual_asr_series(yearly, std_rows)
    for s in series:
        print(f"{s['year']}: ASR {s['asr']:8.2f}")
    e, lo, hi = eapc(series)
    print(f"\nEAPC over full series: {e:+.2f}%/yr (CI {lo:+.2f} to {hi:+.2f})")
    jp, imp = joinpoint_scan(series)
    if jp:
        print(f"Joinpoint detected at {jp} (RSS improved {imp:.0%})")
    else:
        print("No significant joinpoint detected.")

    print()
    print("=" * 64)
    print("Feature 3: bootstrap CI for standardized rate ratio")
    print("=" * 64)
    p1 = _demo_population(scale=1.25, seed=11)
    p2 = _demo_population(scale=1.00, seed=12)
    ratio, bl, bu = bootstrap_rate_ratio_ci(p1, p2, std_rows, n_boot=500)
    print(f"ASR ratio (higher-exposure vs baseline): {ratio:.3f}")
    print(f"95% percentile-bootstrap CI: ({bl:.3f}, {bu:.3f})")
