# HEIC/HEIF Test Fixtures

Place one or more small, valid sample files here for integration tests:

- `*.heic`
- `*.heif`

These files are intentionally not included in-repo by default.

When present, tests in `tests/test_mvp.py` will automatically run fixture-based HEIC checks:

- quality-check decode path (`check_image_quality`)
- CLI project reconstruction workflow wiring with HEIC inputs

Without fixtures, those tests are skipped.
