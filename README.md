# Age Standardized Rate Calculator

> **Domain:** Clinical Decision Support & Biomedical Computing  
> **Reference Guidelines & Standards:** `Standard Clinical Formulations & ISO/IEC Quality Frameworks`

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)
![Audit Trail](https://img.shields.io/badge/Audit-HMAC--SHA256_Tamper--Evident-brightgreen.svg)
![Zero-PHI Guard](https://img.shields.io/badge/Guard-Zero--PHI_Outbound-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)

</div>

---

## 📖 What It Does

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

---

## ⚙️ Key Capabilities & Algorithmic Modules

### 🔬 Core Algorithmic & Evaluation Engines

- **`AgeSpecificData`**: Age-specific counts and person-years.
- **`DirectStandardizationResult`**: Directly age-standardized rate results with multiple confidence intervals.
- **`IndirectStandardizationResult`**: Indirect standardization (SMR/SIR) results.
- **`RateRatioComparisonResult`**: Comparison between two standardized populations (SRR / SRD).
- **`ASRCalculator`**: Master engine for computing ASRs, SMRs, and rate comparisons.

---

## 📐 Mathematical Formulation & Logic

```text
  return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
  return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
  cum_risk = 1.0 - math.exp(-cum_rate)
```

---

## 💻 CLI Quickstart & Usage

### 1. Guided Interactive Mode
```bash
python cli.py
```

### 2. Direct Parameterized Evaluation
```bash
python cli.py --interactive <value> --demo <value> --json <value> --input <value>
```

### Parameter Reference
- `--interactive`: Specifies input measurement or parameter value.
- `--demo`: Specifies input measurement or parameter value.
- `--json`: Specifies input measurement or parameter value.
- `--input`: Specifies input measurement or parameter value.
- `--standard`: Specifies input measurement or parameter value.
- `--alpha`: Specifies input measurement or parameter value.
- `--pop1`: Specifies input measurement or parameter value.
- `--pop2`: Specifies input measurement or parameter value.
- `--name1`: Specifies input measurement or parameter value.
- `--name2`: Specifies input measurement or parameter value.

### Input Data Schema

| Field | Description | Requirement |
|:------|:------------|:------------|
| `suite_name` | Parameter / observation metric | Required |
| `system_slug` | Parameter / observation metric | Required |
| `standard_reference` | Parameter / observation metric | Required |
| `test_cases` | Parameter / observation metric | Required |

---

## 🛡️ Security & Enterprise Architecture

* **Zero-PHI Outbound Interceptor:** Active AST and regex inspection blocking SSNs, MRNs, phone numbers, and patient identifiers.
* **Tamper-Evident HMAC-SHA256 Audit Trail:** Chained, cryptographically signed logs for every evaluation and state transition.
* **Air-Gapped LLM Reasoning Adapter:** Agnostic integration for local Ollama instances (`llama3`, `mistral`), Claude 3.5 Sonnet, GPT-4o, and deterministic test mocks.
* **Active Learning Bayesian Calibration:** Dynamic tracker updating worker reliability weights and monitoring Brier calibration drift.
* **FastAPI & Prometheus Telemetry:** Exposes OpenAPI 3.1 REST endpoints and operational Prometheus metrics (`/metrics`).

---

## 🧪 Testing & Verification

Run the automated test suite:

```bash
pytest -v
```

Execute high-throughput batch simulation benchmarks:

```bash
python simulator.py --tasks 1000 --concurrency 8
```

---

## 🐳 Container Deployment

```bash
docker build -t age-standardized-rate-calculator .
docker run -p 8000:8000 age-standardized-rate-calculator
```
