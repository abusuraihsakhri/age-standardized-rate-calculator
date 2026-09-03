# Age-Standardized Rate (ASR) Calculator

A pure Python epidemiological and demographic statistical engine implementing:
- Direct age-standardization for incidence and mortality rates across population cohorts.
- Fay & Feuer (1997) Gamma-distribution confidence intervals (the gold standard used in NCI SEER*Stat).
- Normal approximation (Wald) and log-transformed confidence intervals.
- Indirect standardization: Standardized Mortality / Incidence Ratio (SMR / SIR) with exact Poisson confidence intervals and Byar's approximation.
- Standardized Rate Ratio (SRR) and Standardized Rate Difference (SRD) with delta-method variance propagation.
- Cumulative incidence rate and cumulative lifetime risk ($0\text{--}74\text{ years}$).
- Built-in standard reference populations: WHO World Standard (2000–2025), US 2000 Standard, Segi 1960, and European Standard Population (ESP 2013).

Requires Python standard library only (zero external runtime dependencies).

---

## Features

- **Direct Standardization:**
  $$\text{ASR} = \sum_{i} w_i \cdot \frac{d_i}{n_i}, \quad \text{Var}(\text{ASR}) = \sum_{i} w_i^2 \cdot \frac{d_i}{n_i^2}$$
- **Fay & Feuer (1997) Gamma Confidence Intervals:**
  Exact gamma-distributed confidence intervals preserving coverage probability even in the presence of sparse event counts and small age-band denominators.
- **Indirect Standardization (SMR / SIR):**
  Calculates expected events based on reference stratum-specific rates and computes two-sided exact Poisson confidence intervals.
- **Population Rate Comparisons:**
  Evaluates statistical significance of rate differences and relative risk ratios between two independent populations.
- **Batch CSV Processing:** High-throughput processing of population surveillance tables.

---

## Installation & Requirements

- Python 3.10+ (tested on 3.10, 3.11, 3.12)
- Zero external runtime dependencies. `pytest` is optional for running tests.

```bash
git clone https://github.com/abusuraihsakhri/age-standardized-rate-calculator.git
cd age-standardized-rate-calculator
```

---

## CLI Usage

### 1. Direct Age-Standardization
Compute ASR from CSV with age-specific counts and person-years:
```bash
python cli.py direct --input sample.csv --standard who2000
```
Output as JSON:
```bash
python cli.py direct --input sample.csv --standard who2000 --json
```

### 2. Compare Two Populations
Compare standardized rates between two regional cohorts:
```bash
python cli.py compare --pop1 examples/population_a.csv --pop2 examples/population_b.csv --name1 "Region North" --name2 "Region South"
```
Output as JSON:
```bash
python cli.py compare --pop1 examples/population_a.csv --pop2 examples/population_b.csv --json
```

### 3. Run Benchmark Suite Demo
```bash
python cli.py --demo
```

### 4. Batch CSV Computation
Export ASR calculations to CSV:
```bash
python cli.py batch --input sample.csv --output results.csv
```

---

## Python API Quickstart

```python
from asr_calculator import (
    AgeSpecificData,
    ASRCalculator,
    direct_standardize,
    fay_feuer_ci,
)

# 1. Load age-specific data
data = AgeSpecificData(
    age_groups=["0-4", "5-9", "10-14"],
    counts=[10.0, 15.0, 20.0],
    person_years=[10000.0, 10000.0, 10000.0],
)

# 2. Compute Direct ASR with WHO 2000 standard population
result = ASRCalculator.calculate_direct_asr(data, standard_population=[("0-4", 8860), ("5-9", 8690), ("10-14", 8600)])
print(f"Crude Rate: {result.crude_rate_per_100k:.2f} per 100k")
print(f"ASR: {result.asr_per_100k:.2f} per 100k")
print(f"Fay & Feuer 95% CI: [{result.fay_feuer_ci_95[0]:.2f} - {result.fay_feuer_ci_95[1]:.2f}]")

# 3. Indirect Standardization (SMR)
ref_rates = {"0-4": 0.001, "5-9": 0.001, "10-14": 0.001}
smr_result = ASRCalculator.calculate_smr(data, ref_rates)
print(f"SMR: {smr_result.smr:.2f} (Exact Poisson 95% CI: {smr_result.exact_poisson_ci_95})")
```

---

## Running Tests

Run the test suite using standard `unittest` or `pytest`:

```bash
python -m unittest discover tests
# or
pytest -v
```
