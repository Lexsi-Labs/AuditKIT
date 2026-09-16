# Contributing

Thank you for your interest in contributing to AuditKIT!

## Development setup

```bash
git clone https://github.com/Lexsi-Labs/AuditKIT.git
cd AuditKIT
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest -q
```

All tests must pass before submitting a pull request.

## Code standards

- Python 3.10+ compatible
- `from __future__ import annotations` in every module
- Zero required third-party deps in the core library; optional extras for heavy backends
- All metrics inherit from `auditkit.metric.Metric`
- All models inherit from `auditkit.model.Model`
- Test files go in `tests/`, one per module or feature area

## Pull request process

1. Fork the repository and create a feature branch
2. Write tests for your changes
3. Run the full test suite (`pytest -q`)
4. Submit a pull request with a clear description of the changes

## Adding a new metric

1. Create a new file in `src/auditkit/metrics/` or add to an existing family file
2. Inherit from `Metric` and implement `score(sample, output, context)`
3. Register the metric name in `__all__` in `src/auditkit/__init__.py`
4. Add tests in `tests/`

## Adding a new model backend

1. Create a new file in `src/auditkit/model/`
2. Inherit from `Model` and implement `generate(requests)`
3. Add the prefix to `_T1_BACKENDS` in `src/auditkit/model/__init__.py`
4. Add tests in `tests/`

## Documentation

Documentation is built with MKDocs and the Material theme:

```bash
pip install mkdocs mkdocstrings mkdocstrings-python
mkdocs serve
```

See existing pages for style conventions.
