# Age-Standardized Rate Calculator

A Python implementation of direct and indirect age standardization, rate comparisons, confidence intervals, and related epidemiological summaries.

## What it does

- Direct age-standardized rates using WHO 2000-2025, US 2000, Segi 1960, or ESP 2013 reference populations.
- Fay & Feuer gamma confidence intervals, Wald intervals, and log-transformed intervals.
- Indirect standardization with SMR/SIR and Poisson confidence intervals.
- Standardized rate ratios and rate differences.
- Cumulative rate and cumulative risk through age 74.
- CSV input, batch output, and a command-line interface.
- Optional NumPy-based multi-standard, time-series, joinpoint-scan, and bootstrap utilities.
- Static browser calculator in `docs/` with local CSV parsing and no server upload.

## Browser application

The browser app is a static implementation of the direct-standardization workflow. It runs entirely in the browser, supports light and dark themes, accepts CSV files locally, and can export the calculated summary as CSV.

## CLI

Python 3.10+ is supported.

```bash
python cli.py direct --input sample.csv --standard who2000
python cli.py direct --input sample.csv --standard who2000 --json
python cli.py compare --pop1 examples/population_a.csv --pop2 examples/population_b.csv
python cli.py batch --input sample.csv --output results.csv
python cli.py --demo
```

## Python API

```python
from asr_calculator import AgeSpecificData, ASRCalculator

data = AgeSpecificData(
    age_groups=["0-4", "5-9", "10-14"],
    counts=[10.0, 15.0, 20.0],
    person_years=[10000.0, 10000.0, 10000.0],
)

result = ASRCalculator.calculate_direct_asr(
    data,
    standard_population=[("0-4", 8860), ("5-9", 8690), ("10-14", 8600)],
)

print(result.asr_per_100k)
print(result.fay_feuer_ci_95)
```

Optional enrichment functions are in `asr_enrichment_features.py`:

```python
from asr_enrichment_features import multi_standard_comparison
```

## CSV format

Accepted aliases are:

- age: `age_group` or `age`
- events: `count`, `cases`, or `events`
- denominator: `person_years`, `population`, or `py`

Example:

```csv
age_group,count,person_years
0-4,1,52000
5-9,1,51000
10-14,2,50000
```

Counts must be finite and non-negative. Person-years must be finite and strictly positive for every row. Duplicate age groups are rejected.

## Tests

```bash
python -m pip install -r requirements.txt
pytest -q
python cli.py direct --input sample.csv --json
```

CI tests Python 3.10, 3.11, and 3.12 and performs a syntax check of the static browser application.

## Privacy

The command-line tool reads local files. The browser application processes entered and uploaded data locally in the browser and does not send them to a server.

## Technology

- Python standard library for the core statistical engine
- NumPy for optional enrichment features
- HTML, CSS, and vanilla JavaScript for the browser calculator
- GitHub Actions for CI and Pages deployment

## License

MIT. See [LICENSE](LICENSE).
