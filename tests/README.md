# Tests

Tests are organized per module, mirroring the `src/auditkit` package structure.
Covers all metric families, runner, CLI, red teaming, and comparison.

Run all tests with:

```bash
pytest -q
```

Run a specific module:

```bash
pytest -q tests/test_metrics.py
```
