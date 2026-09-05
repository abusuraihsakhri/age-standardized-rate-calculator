#!/usr/bin/env python3
"""
Age-Standardized Rate (ASR) Calculator
======================================
A pure Python standard library epidemiological and statistical engine implementing:
- Direct age-standardization for incidence and mortality rates
- Fay & Feuer (1997) Gamma-distribution confidence intervals (SEER*Stat gold standard)
- Normal approximation (Wald) and log-transformed confidence intervals
- Indirect standardization: Standardized Mortality/Incidence Ratio (SMR/SIR) with exact Poisson CIs
- Standardized Rate Ratio (SRR) and Standardized Rate Difference (SRD) with delta-method CIs
- Cumulative rate and cumulative risk (0-74 years)
- Standard population benchmarks: WHO World Standard (2000-2025), Segi 1960, European 2013, US 2000 Standard.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union

__version__ = "2.0.0"


# ============================================================================
# Pure Standard Library Statistical Distributions (Zero-Dependency)
# ============================================================================

def log_gamma(x: float) -> float:
    """Lanczos log-gamma approximation (accurate to 1e-13)."""
    if x <= 0:
        raise ValueError(f"log_gamma requires positive argument, got {x}")
    # math.lgamma is available in Python 3 standard library
    return math.lgamma(x)


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """
    Inverse of the standard normal CDF (quantile function).
    Acklam's algorithm (accurate to within 1.15e-9).
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"Probability must be in (0, 1), got {p}")

    # Coefficients in rational approximations
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]

    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798529320e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]

    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]

    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)


def gamma_inc_lower(a: float, x: float) -> float:
    """Lower regularized incomplete gamma function P(a, x) = gamma(a, x) / Gamma(a)."""
    if x <= 0.0:
        return 0.0
    if a <= 0.0:
        return 1.0

    # Series expansion for x < a + 1
    if x < a + 1.0:
        ap = a
        sum_val = 1.0 / a
        del_val = sum_val
        for _ in range(100):
            ap += 1.0
            del_val *= x / ap
            sum_val += del_val
            if abs(del_val) < abs(sum_val) * 1e-15:
                break
        return sum_val * math.exp(-x + a * math.log(x) - log_gamma(a))
    else:
        # Continued fraction approximation for upper incomplete gamma Q(a, x)
        b = x + 1.0 - a
        c = 1.0 / 1e-30
        d = 1.0 / b
        h = d
        for i in range(1, 100):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < 1e-30: d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30: c = 1e-30
            d = 1.0 / d
            del_val = d * c
            h *= del_val
            if abs(del_val - 1.0) < 1e-15:
                break
        q_val = math.exp(-x + a * math.log(x) - log_gamma(a)) * h
        return max(0.0, min(1.0, 1.0 - q_val))


def chi2_cdf(x: float, df: float) -> float:
    """Chi-Square cumulative distribution function F(x; df)."""
    if x <= 0.0:
        return 0.0
    return gamma_inc_lower(df / 2.0, x / 2.0)


def chi2_ppf(p: float, df: float) -> float:
    """
    Inverse Chi-Square CDF (percent point function) with Newton-Raphson refinement.
    """
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return float("inf")
    if df <= 0.0:
        return 0.0

    # Wilson-Hilferty starting estimate
    z = normal_ppf(p)
    w = 2.0 / (9.0 * df)
    term = 1.0 - w + z * math.sqrt(w)
    x = df * (term ** 3) if term > 0 else df * math.exp(z * math.sqrt(2.0 / df))
    x = max(1e-8, x)

    # Newton-Raphson refinement
    for _ in range(12):
        cdf_val = chi2_cdf(x, df)
        err = cdf_val - p
        if abs(err) < 1e-12:
            break
        # PDF f(x; df) = (1 / (2^(df/2) * Gamma(df/2))) * x^(df/2 - 1) * exp(-x/2)
        try:
            log_pdf = - (df / 2.0) * math.log(2.0) - log_gamma(df / 2.0) + (df / 2.0 - 1.0) * math.log(x) - x / 2.0
            pdf_val = math.exp(log_pdf)
            if pdf_val > 1e-15:
                step = err / pdf_val
                x = max(1e-10, x - step)
        except (ValueError, OverflowError):
            break

    return x


# ============================================================================
# Standard Reference Populations
# ============================================================================

# WHO World Standard Population 2000-2025 (Ahmad et al. 2001, WHO)
WHO_WORLD_2000_2025: List[Tuple[str, float]] = [
    ("0-4", 8860.0), ("5-9", 8690.0), ("10-14", 8600.0), ("15-19", 8470.0),
    ("20-24", 8220.0), ("25-29", 7930.0), ("30-34", 7610.0), ("35-39", 7150.0),
    ("40-44", 6590.0), ("45-49", 6040.0), ("50-54", 5370.0), ("55-59", 4550.0),
    ("60-64", 3720.0), ("65-69", 2960.0), ("70-74", 2210.0), ("75-79", 1520.0),
    ("80-84", 910.0), ("85+", 635.0),
]

# US 2000 Standard Population (18 age groups, NCI SEER)
US_2000_STANDARD: List[Tuple[str, float]] = [
    ("0-4", 18987000.0), ("5-9", 19920000.0), ("10-14", 20202000.0),
    ("15-19", 20092000.0), ("20-24", 19928000.0), ("25-29", 19703000.0),
    ("30-34", 20437000.0), ("35-39", 22842000.0), ("40-44", 22933000.0),
    ("45-49", 19983000.0), ("50-54", 17562000.0), ("55-59", 13434000.0),
    ("60-64", 10733000.0), ("65-69", 9480000.0), ("70-74", 8857000.0),
    ("75-79", 7415000.0), ("80-84", 4945000.0), ("85+", 4259000.0),
]

# Segi 1960 World Standard
SEGI_1960_STANDARD: List[Tuple[str, float]] = [
    ("0-4", 12000.0), ("5-9", 10000.0), ("10-14", 9000.0), ("15-19", 8000.0),
    ("20-24", 8000.0), ("25-29", 6000.0), ("30-34", 6000.0), ("35-39", 6000.0),
    ("40-44", 6000.0), ("45-49", 6000.0), ("50-54", 5000.0), ("55-59", 4000.0),
    ("60-64", 4000.0), ("65-69", 3000.0), ("70-74", 2000.0), ("75-79", 1000.0),
    ("80-84", 500.0), ("85+", 500.0),
]

# European Standard Population (ESP 2013)
EUROPEAN_2013_STANDARD: List[Tuple[str, float]] = [
    ("0-4", 5000.0), ("5-9", 5500.0), ("10-14", 5500.0), ("15-19", 5500.0),
    ("20-24", 6000.0), ("25-29", 6000.0), ("30-34", 6500.0), ("35-39", 7000.0),
    ("40-44", 7000.0), ("45-49", 7000.0), ("50-54", 7000.0), ("55-59", 6500.0),
    ("60-64", 6000.0), ("65-69", 5500.0), ("70-74", 5000.0), ("75-79", 4000.0),
    ("80-84", 2500.0), ("85+", 1500.0),
]

BUILTIN_STANDARDS: Dict[str, List[Tuple[str, float]]] = {
    "who2000": WHO_WORLD_2000_2025,
    "us2000": US_2000_STANDARD,
    "segi1960": SEGI_1960_STANDARD,
    "european2013": EUROPEAN_2013_STANDARD,
}


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class AgeSpecificData:
    """Age-specific counts and person-years."""
    age_groups: List[str]
    counts: List[float]
    person_years: List[float]

    def __post_init__(self):
        if len(self.age_groups) != len(self.counts) or len(self.counts) != len(self.person_years):
            raise ValueError("Lengths of age_groups, counts, and person_years must match.")
        if any(c < 0 for c in self.counts):
            raise ValueError("Event counts cannot be negative.")
        if any(py < 0 for py in self.person_years):
            raise ValueError("Person-years cannot be negative.")

    def total_events(self) -> float:
        return sum(self.counts)

    def total_person_years(self) -> float:
        return sum(self.person_years)

    def crude_rate(self) -> float:
        tot_py = self.total_person_years()
        if tot_py <= 0:
            raise ValueError("Total person-years must be positive.")
        return self.total_events() / tot_py


@dataclass
class DirectStandardizationResult:
    """Directly age-standardized rate results with multiple confidence intervals."""
    standard_name: str
    crude_rate_per_100k: float
    asr_per_100k: float
    standard_error_per_100k: float
    variance: float
    fay_feuer_ci_95: Tuple[float, float]
    wald_ci_95: Tuple[float, float]
    log_transformed_ci_95: Tuple[float, float]
    cumulative_rate_0_74_pct: float
    cumulative_risk_0_74_pct: float
    total_events: float
    total_person_years: float
    age_group_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class IndirectStandardizationResult:
    """Indirect standardization (SMR/SIR) results."""
    observed_events: float
    expected_events: float
    smr: float
    exact_poisson_ci_95: Tuple[float, float]
    byar_ci_95: Tuple[float, float]
    p_value_vs_unity: float
    interpretation: str


@dataclass
class RateRatioComparisonResult:
    """Comparison between two standardized populations (SRR / SRD)."""
    population_1_name: str
    population_2_name: str
    asr_1_per_100k: float
    asr_2_per_100k: float
    rate_ratio: float
    rate_ratio_ci_95: Tuple[float, float]
    rate_difference_per_100k: float
    rate_difference_ci_95: Tuple[float, float]
    two_sided_p_value: float
    statistically_significant: bool


# ============================================================================
# Core Standardization Algorithms
# ============================================================================

def direct_standardize(
    weights: Sequence[float],
    counts: Sequence[float],
    person_years: Sequence[float],
) -> Tuple[float, float]:
    """
    Direct standardization:
    ASR = sum_i (w_i * (d_i / n_i))
    Var(ASR) = sum_i (w_i^2 * d_i / n_i^2)
    """
    if len(weights) != len(counts) or len(counts) != len(person_years):
        raise ValueError("Lengths of weights, counts, and person_years must match.")
    w_sum = sum(weights)
    if w_sum <= 0:
        raise ValueError("Sum of weights must be positive.")
    norm_w = [w / w_sum for w in weights]

    asr = 0.0
    var = 0.0
    for w, d, n in zip(norm_w, counts, person_years):
        if n > 0:
            rate = d / n
            asr += w * rate
            var += (w ** 2) * d / (n ** 2)
        elif d > 0:
            raise ValueError("Cannot have positive events with zero person-years.")
    return asr, var


def fay_feuer_ci(
    asr_val: float,
    var_val: float,
    weights: Sequence[float],
    person_years: Sequence[float],
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Fay & Feuer (1997) Gamma-distribution confidence intervals.
    Standard algorithm used in NCI SEER*Stat.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    w_sum = sum(weights)
    norm_w = [w / w_sum for w in weights]

    wm_candidates = [(w / n) for w, n in zip(norm_w, person_years) if n > 0]
    wm = max(wm_candidates) if wm_candidates else 0.0

    if asr_val == 0.0 or var_val == 0.0:
        lower = 0.0
        df_u = 2.0 * (wm ** 2) / (wm ** 2)
        upper = (wm / 2.0) * chi2_ppf(1.0 - alpha / 2.0, df_u)
        return lower, upper

    df_l = 2.0 * (asr_val ** 2) / var_val
    lower = (var_val / (2.0 * asr_val)) * chi2_ppf(alpha / 2.0, df_l)

    df_u = 2.0 * ((asr_val + wm) ** 2) / (var_val + wm ** 2)
    upper = ((var_val + wm ** 2) / (2.0 * (asr_val + wm))) * chi2_ppf(1.0 - alpha / 2.0, df_u)

    return lower, upper


def indirect_standardize(
    data: AgeSpecificData,
    ref_rates: Dict[str, float],
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """
    Indirect standardization:
    E = sum_i (ref_rate_i * n_i)
    SMR = O / E
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    observed = data.total_events()
    expected = 0.0
    for age, py in zip(data.age_groups, data.person_years):
        if age in ref_rates:
            expected += ref_rates[age] * py
        else:
            raise KeyError(f"Missing reference rate for age group: {age}")

    if expected <= 0.0:
        raise ValueError("Expected events must be positive for indirect standardization.")

    smr = observed / expected
    return observed, expected, smr


def smr_poisson_ci(
    observed: float,
    expected: float,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Exact Poisson confidence intervals for SMR."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if expected <= 0.0:
        raise ValueError("Expected events must be positive.")

    if observed == 0.0:
        lower = 0.0
        upper = chi2_ppf(1.0 - alpha / 2.0, 2.0) / (2.0 * expected)
    else:
        lower = chi2_ppf(alpha / 2.0, 2.0 * observed) / (2.0 * expected)
        upper = chi2_ppf(1.0 - alpha / 2.0, 2.0 * (observed + 1.0)) / (2.0 * expected)

    return lower, upper


# ============================================================================
# High-Level Age-Standardized Rate Engine
# ============================================================================

class ASRCalculator:
    """Master engine for computing ASRs, SMRs, and rate comparisons."""

    @classmethod
    def calculate_direct_asr(
        cls,
        data: AgeSpecificData,
        standard_population: Union[str, List[Tuple[str, float]]] = "who2000",
        alpha: float = 0.05,
        multiplier: float = 100000.0,
    ) -> DirectStandardizationResult:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if multiplier <= 0:
            raise ValueError(f"multiplier must be positive, got {multiplier}")
        if isinstance(standard_population, str):
            std_name = standard_population
            std_pop = BUILTIN_STANDARDS.get(standard_population.lower())
            if not std_pop:
                raise ValueError(f"Unknown built-in standard: {standard_population}. Choose from {list(BUILTIN_STANDARDS.keys())}")
        else:
            std_name = "Custom Standard"
            std_pop = standard_population

        std_dict = dict(std_pop)
        matched_weights = []
        matched_counts = []
        matched_py = []
        age_details = []

        # Cumulative rate (0-74) accumulator (assuming 5-year bands up to 74)
        cum_rate = 0.0

        for age, count, py in zip(data.age_groups, data.counts, data.person_years):
            if age in std_dict:
                w = std_dict[age]
                matched_weights.append(w)
                matched_counts.append(count)
                matched_py.append(py)

                r_i = (count / py) if py > 0 else 0.0
                age_details.append({
                    "age_group": age,
                    "count": count,
                    "person_years": py,
                    "age_specific_rate_per_100k": round(r_i * multiplier, 2),
                    "standard_weight": w,
                })

                # If age is under 75 years
                if any(tag in age for tag in ["0-", "5-", "10-", "15-", "20-", "25-", "30-", "35-", "40-", "45-", "50-", "55-", "60-", "65-", "70-"]):
                    cum_rate += 5.0 * r_i
            else:
                raise ValueError(f"Age group '{age}' from data not found in standard population.")

        asr, var = direct_standardize(matched_weights, matched_counts, matched_py)
        se = math.sqrt(var)

        # Confidence intervals
        ff_low, ff_up = fay_feuer_ci(asr, var, matched_weights, matched_py, alpha=alpha)

        # Wald normal interval
        z = normal_ppf(1.0 - alpha / 2.0)
        wald_low = max(0.0, asr - z * se)
        wald_up = asr + z * se

        # Log interval
        if asr > 0:
            log_factor = math.exp(z * se / asr)
            log_low = asr / log_factor
            log_up = asr * log_factor
        else:
            log_low, log_up = 0.0, 0.0

        crude = data.crude_rate()
        cum_risk = 1.0 - math.exp(-cum_rate)

        return DirectStandardizationResult(
            standard_name=std_name,
            crude_rate_per_100k=round(crude * multiplier, 2),
            asr_per_100k=round(asr * multiplier, 2),
            standard_error_per_100k=round(se * multiplier, 4),
            variance=var,
            fay_feuer_ci_95=(round(ff_low * multiplier, 2), round(ff_up * multiplier, 2)),
            wald_ci_95=(round(wald_low * multiplier, 2), round(wald_up * multiplier, 2)),
            log_transformed_ci_95=(round(log_low * multiplier, 2), round(log_up * multiplier, 2)),
            cumulative_rate_0_74_pct=round(cum_rate * 100.0, 2),
            cumulative_risk_0_74_pct=round(cum_risk * 100.0, 2),
            total_events=data.total_events(),
            total_person_years=data.total_person_years(),
            age_group_details=age_details,
        )

    @classmethod
    def calculate_smr(
        cls,
        data: AgeSpecificData,
        ref_rates_per_py: Dict[str, float],
        alpha: float = 0.05,
    ) -> IndirectStandardizationResult:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        obs, exp, smr = indirect_standardize(data, ref_rates_per_py, alpha=alpha)
        p_low, p_up = smr_poisson_ci(obs, exp, alpha=alpha)

        # Byar approximation
        z = normal_ppf(1.0 - alpha / 2.0)
        byar_l = (obs * (1.0 - 1.0 / (9.0 * obs) - (z / 3.0) * math.sqrt(1.0 / obs)) ** 3) / exp if obs > 0 else 0.0
        byar_u = ((obs + 1.0) * (1.0 - 1.0 / (9.0 * (obs + 1.0)) + (z / 3.0) * math.sqrt(1.0 / (obs + 1.0))) ** 3) / exp if obs > 0 else 0.0

        # Two-sided Poisson p-value vs unity
        z_stat = (obs - exp) / math.sqrt(exp)
        p_val = 2.0 * (1.0 - normal_cdf(abs(z_stat)))

        if smr > 1.0 and p_val < 0.05:
            interp = f"Statistically significant excess risk (SMR = {smr:.2f}, {int((smr-1)*100)}% elevation over reference)."
        elif smr < 1.0 and p_val < 0.05:
            interp = f"Statistically significant deficit in risk (SMR = {smr:.2f}, {int((1-smr)*100)}% reduction vs reference)."
        else:
            interp = f"No statistically significant difference from reference population (SMR = {smr:.2f}, p = {p_val:.3f})."

        return IndirectStandardizationResult(
            observed_events=obs,
            expected_events=round(exp, 2),
            smr=round(smr, 3),
            exact_poisson_ci_95=(round(p_low, 3), round(p_up, 3)),
            byar_ci_95=(round(byar_l, 3), round(byar_u, 3)),
            p_value_vs_unity=round(p_val, 4),
            interpretation=interp,
        )

    @classmethod
    def compare_standardized_rates(
        cls,
        res1: DirectStandardizationResult,
        res2: DirectStandardizationResult,
        pop1_name: str = "Population 1",
        pop2_name: str = "Population 2",
        alpha: float = 0.05,
    ) -> RateRatioComparisonResult:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        asr1 = res1.asr_per_100k
        asr2 = res2.asr_per_100k
        var1 = (res1.standard_error_per_100k) ** 2
        var2 = (res2.standard_error_per_100k) ** 2

        if asr2 <= 0:
            raise ValueError("Comparison population ASR must be positive to compute rate ratio.")

        ratio = asr1 / asr2
        # Delta method for SE(log ratio)
        se_log_ratio = math.sqrt((var1 / (asr1 ** 2)) + (var2 / (asr2 ** 2))) if asr1 > 0 else 0.0
        z = normal_ppf(1.0 - alpha / 2.0)
        ci_ratio = (round(ratio * math.exp(-z * se_log_ratio), 3), round(ratio * math.exp(z * se_log_ratio), 3))

        diff = asr1 - asr2
        se_diff = math.sqrt(var1 + var2)
        ci_diff = (round(diff - z * se_diff, 2), round(diff + z * se_diff, 2))

        # Test statistic
        z_stat = diff / se_diff if se_diff > 0 else 0.0
        p_val = 2.0 * (1.0 - normal_cdf(abs(z_stat)))

        return RateRatioComparisonResult(
            population_1_name=pop1_name,
            population_2_name=pop2_name,
            asr_1_per_100k=asr1,
            asr_2_per_100k=asr2,
            rate_ratio=round(ratio, 3),
            rate_ratio_ci_95=ci_ratio,
            rate_difference_per_100k=round(diff, 2),
            rate_difference_ci_95=ci_diff,
            two_sided_p_value=round(p_val, 4),
            statistically_significant=p_val < alpha,
        )


def read_age_specific_csv(path: str) -> AgeSpecificData:
    """Reads a CSV with columns: age_group, count/cases/events, person_years/population.

    Validates that the file exists, has a .csv extension, and contains
    non-negative numeric values for counts and person-years.
    """
    # Security: validate file extension and existence
    csv_path = Path(path)
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    age_groups, counts, person_years = [], [], []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):  # row 1 is header
            # Flexible column lookup
            age = row.get("age_group", row.get("age", "")).strip()
            if not age:
                raise ValueError(f"Missing age_group in row {row_num} of {path}")
            try:
                cnt = float(row.get("count", row.get("cases", row.get("events", 0.0))))
                py = float(row.get("person_years", row.get("population", row.get("py", 0.0))))
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid numeric value in row {row_num} of {path}: {e}")
            if cnt < 0:
                raise ValueError(f"Negative count in row {row_num} of {path}: {cnt}")
            if py < 0:
                raise ValueError(f"Negative person-years in row {row_num} of {path}: {py}")
            age_groups.append(age)
            counts.append(cnt)
            person_years.append(py)
    if not age_groups:
        raise ValueError(f"No valid data rows found in {path}")
    return AgeSpecificData(age_groups, counts, person_years)


# ============================================================================
# Helper Functions for Enrichment Features & External Modules
# ============================================================================

def standard_weights(std_rows: Sequence[Tuple[str, float]]) -> Tuple[List[str], List[float]]:
    """
    Extract age group labels and proportional weights from a standard population table.

    Returns (age_groups, weights) where weights sum to 1.0.
    """
    if not std_rows:
        raise ValueError("Standard population rows cannot be empty.")
    ages = [r[0] for r in std_rows]
    raw_weights = [float(r[1]) for r in std_rows]
    total = sum(raw_weights)
    if total <= 0:
        raise ValueError("Sum of standard population weights must be positive.")
    weights = [w / total for w in raw_weights]
    return ages, weights


def align_to_standard(
    data: AgeSpecificData,
    std_ages: Sequence[str],
    std_weights: Sequence[float],
) -> Tuple[List[float], List[float], List[float]]:
    """
    Align age-specific data to a standard population's age bands.

    Returns (matched_weights, matched_counts, matched_person_years) for age
    groups present in both data and standard. Raises ValueError if an age
    group in the standard is missing from the data.
    """
    data_dict = dict(zip(data.age_groups, zip(data.counts, data.person_years)))
    matched_w, matched_c, matched_py = [], [], []
    for age, w in zip(std_ages, std_weights):
        if age not in data_dict:
            raise ValueError(f"Age group '{age}' from standard not found in data.")
        c, py = data_dict[age]
        matched_w.append(w)
        matched_c.append(c)
        matched_py.append(py)
    return matched_w, matched_c, matched_py


def direct_standardization_report(
    data: AgeSpecificData,
    std_rows: Sequence[Tuple[str, float]],
    alpha: float = 0.05,
    per: float = 100000.0,
) -> Dict[str, float]:
    """
    Compute a direct age-standardization report (compatible with enrichment features).

    Returns a dict with keys:
        - asr: age-standardized rate (scaled by `per`)
        - asr_raw: age-standardized rate (per person-year, unscaled)
        - asr_lower: lower confidence bound (Fay & Feuer)
        - asr_upper: upper confidence bound (Fay & Feuer)
        - crude_rate: crude rate (scaled by `per`)
    """
    ages, weights = standard_weights(std_rows)
    w, c, py = align_to_standard(data, ages, weights)
    asr_raw, var_raw = direct_standardize(weights, c, py)
    low, high = fay_feuer_ci(asr_raw, var_raw, weights, py, alpha=alpha)
    crude = data.crude_rate()
    return {
        "asr": round(asr_raw * per, 2),
        "asr_raw": asr_raw,
        "asr_lower": round(low * per, 2),
        "asr_upper": round(high * per, 2),
        "crude_rate": round(crude * per, 2),
    }
