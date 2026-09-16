# Development

## Setup

```bash
git clone https://github.com/Lexsi-Labs/AuditKIT
cd AuditKIT
python3 -m pytest tests/ -q
```

## Running Tests

```bash
# Full suite (750+ tests)
python3 -m pytest tests/

# Specific test file
python3 -m pytest tests/test_integration.py -v

# With coverage
pip install pytest-cov
python3 -m pytest tests/ --cov=src/auditkit
```

## Test Structure

| File | Tests |
|---|---|
| `tests/test_integration.py` | End-to-end pipeline tests |
| `tests/test_api.py` | API tests (evaluate, compare) |
| `tests/test_runner.py` | Runner tests |
| `tests/test_generation_metrics.py` | Generation metric tests |
| `tests/test_toxicity_metrics.py` | Toxicity metric tests |
| `tests/test_pairwise_metrics.py` | Pairwise metric tests |
| `tests/test_hallucination_metrics.py` | Hallucination metric tests |
| ... | ... |

## Building

```bash
pip install build
python3 -m build --wheel
```

## Publishing

```bash
pip install twine
python3 -m twine upload dist/*
```

## CI

No GitHub Actions workflow exists in this repo yet — add one at
`.github/workflows/ci.yml` to run tests on push/PR.
