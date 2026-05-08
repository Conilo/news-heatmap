## Summary

<!-- What does this PR do and why? -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / cleanup
- [ ] Chore (deps, CI, docs)

## Testing done locally

- [ ] `pytest` — fast unit tests pass
- [ ] `pytest tests/test_extract_governor.py --slm-live` — run if `src/extract.py` was modified
- [ ] `pytest --live` — run if `src/fetch.py` was modified

## Checklist

- [ ] CI unit tests pass (required for merge)
- [ ] Governor SLM test passes if `src/extract.py` was changed
- [ ] Live fetch test passes if `src/fetch.py` was changed
- [ ] `README.md` updated if behavior or usage changed
- [ ] Fixture CSV (`tests/fixtures/slm_eval/`) updated if new cases were added
