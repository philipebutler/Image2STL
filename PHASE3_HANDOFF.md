# Phase 3 Handoff — Reconstruction Engine with Fallback

**Date:** 2026-02-28
**Phase:** 3 of 6
**Status:** ✅ Complete

---

## What Was Delivered

### New Files

```
core/reconstruction/
└── engine.py              # ReconstructionEngine + MethodAttempt
```

### Modified Files

| File | Change |
|---|---|
| `core/reconstruction/__init__.py` | Exports `ReconstructionEngine`, `MethodAttempt` |
| `core/reconstruction_engine.py` | Added `reconstruct_multi()` method |

### New Test File

- **File:** `tests/test_reconstruction_engine.py`
- **Count:** 25 new tests, all passing (105 total including Phases 1 & 2)

---

## Key Classes

### `MethodAttempt` (dataclass)

```python
@dataclass
class MethodAttempt:
    method: ReconstructionMethod
    result: ReconstructionResult
```

Records each reconstruction attempt within a run.  Accessible via
`engine.attempts` after `reconstruct()` returns.

### `ReconstructionEngine` (in `core/reconstruction/engine.py`)

| Method | Description |
|---|---|
| `reconstruct(images, output_dir, method_chain, ...)` | Main orchestration loop |
| `_get_reconstructor(method)` | Instantiates the right method class |
| `_post_process(mesh_path, output_dir, on_progress)` | repair → optimize → scale → export |
| `_repair(src, dst)` | Stub — returns `src` until Phase 5 |
| `_optimize(src, dst)` | Stub — returns `src` until Phase 5 |
| `_scale_and_export(src, dst, scale_mm)` | Stub — returns `src` until Phase 5 |
| `_build_all_failed_result()` | Consolidated failure result |

#### Fallback loop logic

1. If `method_chain` is `None`, auto-select via `MethodSelector.detect_hardware()`.
2. For each method in the chain:
   - Call `can_run()` — skip if `False` (not recorded in `attempts`).
   - Call `reconstruct()` — record attempt in `attempts`.
   - On success: run `_post_process()` and return.
   - On failure: log, fire `on_method_completed(name, False)`, continue.
3. If all methods exhausted: return `_build_all_failed_result()`.

#### Callbacks (all optional)

| Parameter | Signature | When fired |
|---|---|---|
| `on_progress` | `(percent: int, status: str)` | Wired into each method + post-processing |
| `on_method_started` | `(method_name: str)` | Before each attempt |
| `on_method_completed` | `(method_name: str, success: bool)` | After each attempt |

### `reconstruct_multi()` (in `core/reconstruction_engine.py`)

Wraps `core/reconstruction/engine.py`'s `ReconstructionEngine.reconstruct()` in the
existing daemon-thread model.  Fires `on_success(path, stats)` / `on_error(code, msg)`
consistent with the existing `reconstruct()` / `repair()` / `scale()` API.
Respects `is_running` guard (ignores call if already active).

---

## Design Decisions

1. **No Qt dependency in `core/reconstruction/engine.py`** — callbacks instead of
   signals, keeping the engine independently testable without a display.
2. **Post-processing stubs** — `_repair`, `_optimize`, `_scale_and_export` are
   overridable instance methods.  Phase 5 subclasses (or monkey-patches in tests)
   can replace them without touching the pipeline order.
3. **Skipped methods not recorded in `attempts`** — only methods that actually ran
   (i.e. `can_run()` returned `True`) appear in the attempt history.
4. **`_post_process` takes `on_progress`** — allows progress events at 90/93/96/100%
   during post-processing without duplicating callback wiring.
5. **`_build_all_failed_result` uses friendly names** — reads `get_method_requirements()`
   so error messages say "TripoSR Fusion" not "triposr_fusion".

---

## Tests

### Coverage

| Class | Scenarios |
|---|---|
| `MethodAttempt` | stores method + result (1) |
| Successful first method | returns success, 1 attempt recorded, second method not called (3) |
| Fallback | falls back when first fails, attempts record both, skips non-runnable (3) |
| All methods failed | returns failure result, error message contains each method, no-runnable message (3) |
| Post-processing | called on success, not called on failure, step order, final path, propagated to result (5) |
| Callbacks | `on_method_started`, `on_method_completed` success/failure, `on_progress` wired (4) |
| Package imports | `ReconstructionEngine`, `MethodAttempt` from package root (2) |
| `reconstruct_multi` | exists, `on_success` fires, `on_error` fires, is_running guard (4) |

### Running Tests

```bash
# Phase 3 only
python -m pytest tests/test_reconstruction_engine.py -v

# Phase 1 + 2 + 3
python -m pytest tests/test_reconstruction_foundation.py tests/test_reconstruction_methods.py tests/test_reconstruction_engine.py -v

# Full suite (expect 1 pre-existing failure in test_mvp.py)
python -m pytest -q
```

---

## What Comes Next (Phase 4)

### Objective

Update the desktop UI to support method selection, hardware info display, and
multi-stage progress tracking.

### Files to Create / Modify

- `ui/widgets/control_panel.py` — method selection radio buttons (Auto/E/D/C/Cloud),
  grey out unavailable methods, show estimated time
- `ui/widgets/method_status_widget.py` — NEW: shows which method is running +
  attempt history
- `ui/dialogs/hardware_info_dialog.py` — NEW: detected hardware capabilities dialog
- `ui/widgets/progress_widget.py` — multi-stage progress (method attempt +
  post-processing steps)
- Wire new signals: `method_started`, `method_completed`, `reconstruction_completed/failed`

### Notes for Phase 4

- Use `MethodSelector.detect_hardware()` in the UI constructor (once, cache result).
- Disable radio buttons for methods whose `can_run()` returns `False`.
- Connect `on_method_started` / `on_method_completed` from `reconstruct_multi()` to
  `method_status_widget` to show live attempt history.
- Use `on_progress` (0–100 int) for the progress bar.

---

## Pre-existing Issues (Not Addressed)

- `tests/test_mvp.py::MVPTests::test_scale_command_uses_input_dimensions` — `KeyError: 'scaleFactor'`
  (pre-existing, unrelated to reconstruction refactor)
- 19 tests skipped (optional dependencies not installed in CI environment)

---

## Security Summary

CodeQL analysis: **0 alerts** — no security vulnerabilities introduced.
