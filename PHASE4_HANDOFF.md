# Phase 4 Handoff — UI Integration

**Date:** 2026-02-28
**Phase:** 4 of 6
**Status:** ✅ Complete

---

## What Was Delivered

### New Files

```
ui/widgets/method_status_widget.py   # NEW: shows active method + attempt history
ui/dialogs/hardware_info_dialog.py   # NEW: hardware capabilities dialog
tests/test_ui_phase4.py              # 31 tests (3 non-Qt pass; 28 skip in headless CI)
```

### Modified Files

| File | Change |
|---|---|
| `ui/widgets/control_panel.py` | Replaced Local/Cloud radios with 5-radio method group; added `selected_method` property; hardware-aware enable/disable; `hardware_info_requested` Signal; HW Info button |
| `ui/widgets/progress_widget.py` | Added `set_stage_label` / `clear_stage_label` for multi-stage indication; `reset()` and `set_complete()` clear the stage label |
| `ui/main_window.py` | Imports new widgets/dialogs; adds `MethodStatusWidget` to layout; Reconstruction → Hardware Info… menu; wires `hardware_info_requested`; `_start_reconstruction` branches on cloud vs multi-method; new callbacks `_on_multi_engine_progress`, `_on_method_started`, `_on_method_completed`; `ALL_METHODS_FAILED` error suggestion |

---

## Key Classes / APIs

### `MethodStatusWidget` (`ui/widgets/method_status_widget.py`)

| Method | Description |
|---|---|
| `set_current_method(name)` | Shows "⟳ name" while the method is running |
| `record_attempt(name, success)` | Records outcome; adds a coloured badge to the history row |
| `reset()` | Clears label and attempt history |

### `HardwareInfoDialog` (`ui/dialogs/hardware_info_dialog.py`)

Opened via **Reconstruction → Hardware Info…** or the **HW Info…** button on the control panel.  
Constructed with a `HardwareCapabilities` instance (cached from `MethodSelector.detect_hardware()` at `ControlPanel.__init__` time).

### `ControlPanel` changes

#### New signals / properties

| Name | Type | Description |
|---|---|---|
| `hardware_info_requested` | `Signal()` | Fired by the *HW Info…* button |
| `reconstruction_mode` | `str` property | Returns `"auto"`, `"method_e"`, `"method_d"`, `"method_c"`, or `"cloud"` |
| `selected_method` | `Optional[ReconstructionMethod]` property | `None` when *Auto* is active |

#### Greyed-out methods
- **Method E** radio — disabled when `hardware.can_run_method_e` is `False` (< 6 GB VRAM)
- **Method D** radio — disabled when `hardware.can_run_method_d` is `False` (< 4 GB VRAM)
- Both remain disabled after `set_processing(False)` if hardware constraints apply

#### Tooltips
Each radio button shows: method display name, one-line description, and estimated time in minutes.

### `ProgressWidget` additions

| Method | Description |
|---|---|
| `set_stage_label(text)` | Shows italic grey sub-label; hidden when `text` is empty |
| `clear_stage_label()` | Hides the stage label |
| Stage label auto-cleared by | `set_complete()`, `reset()` |

### `MainWindow` changes

#### Layout (bottom of central widget, vertical order)
1. `ControlPanel`
2. `MethodStatusWidget` (hidden initially; shown only during non-cloud reconstruction)
3. `ProgressWidget`

#### New menu
**Reconstruction → Hardware Info…** — opens `HardwareInfoDialog`.

#### Reconstruction routing in `_start_reconstruction`
- `mode == "cloud"` → calls `reconstruction_engine.reconstruct()` (unchanged legacy path)
- Any other mode → calls `reconstruction_engine.reconstruct_multi()`:
  - *Auto*: passes `method_chain=None` so the engine auto-selects via hardware detection
  - *Method E/D/C*: passes a `method_chain` built by `MethodSelector.select_method(hw, user_preference=…)`

#### Thread-safe callbacks (all dispatch via `QMetaObject.invokeMethod`)
| Callback | Handler slot | Action |
|---|---|---|
| `on_multi_engine_progress(percent, status)` | `_update_progress` | Converts percent→fraction, updates progress bar |
| `on_method_started(method_name)` | `_handle_method_started` | Updates `MethodStatusWidget`; sets stage label |
| `on_method_completed(method_name, success)` | `_handle_method_completed` | Records attempt in widget; on failure sets stage label |

---

## Design Decisions

1. **Hardware detection cached at `ControlPanel` construction** — avoids repeated GPU probing on every UI event; accessible as `self._hardware`.
2. **`MethodStatusWidget` hidden during cloud mode** — the old cloud path doesn't fire method callbacks, so the widget is hidden to avoid visual confusion.
3. **Stage label on `ProgressWidget`, method badges on `MethodStatusWidget`** — separation of concerns: the progress bar area shows *what step*, the method widget shows *which method + history*.
4. **`set_processing(False)` respects hardware constraints** — Method E/D radios re-enable only when hardware allows, preventing user from selecting unavailable methods after processing ends.
5. **`ALL_METHODS_FAILED` error suggestion added** — informs user to try Cloud mode when all local methods exhaust.

---

## Tests

### Coverage

| Class | Scenarios |
|---|---|
| `TestMethodSelectorLogic` | auto chain high vram, user preference first, auto chain no GPU (3) |
| `TestControlPanelMethodSelection` | default auto, cloud/C mode, selected_method, E/D disabled without GPU, set_processing, E enabled with high VRAM (12, skip in headless) |
| `TestProgressWidgetStageLabel` | hidden by default, set/clear/empty/set_complete/reset (6, skip in headless) |
| `TestMethodStatusWidget` | initial hidden, set_current, record success/failure, multiple attempts, reset (6, skip in headless) |
| `TestHardwareInfoDialog` | constructs, title, stores capabilities, importable (4, skip in headless) |

### Running Tests

```bash
# Phase 4 only
python -m pytest tests/test_ui_phase4.py -v

# Phase 1 + 2 + 3 + 4
python -m pytest tests/test_reconstruction_foundation.py tests/test_reconstruction_methods.py tests/test_reconstruction_engine.py tests/test_ui_phase4.py -v

# Full suite (expect 1 pre-existing failure in test_mvp.py)
python -m pytest -q
```

---

## What Comes Next (Phase 5)

### Objective

Replace method stubs with real reconstruction logic using actual AI models.

### Files to Create / Modify

- `core/reconstruction/methods/method_e_hybrid.py` — integrate SyncDreamer + COLMAP
- `core/reconstruction/methods/method_d_dust3r.py` — integrate Dust3R model
- `core/reconstruction/methods/method_c_triposr.py` — integrate TripoSR model
- `core/reconstruction/components/view_synthesizer.py` — SyncDreamer/Zero123 integration
- `core/reconstruction/components/colmap_wrapper.py` — COLMAP subprocess management
- `core/reconstruction/components/mesh_aligner.py` — ICP alignment (trimesh/Open3D)
- `core/reconstruction/components/mesh_verifier.py` — reprojection quality scoring
- `core/reconstruction/engine.py` — implement `_repair`, `_optimize`, `_scale_and_export` stubs

### Notes for Phase 5

- Use `requirements.txt` to declare new dependencies (torch, dust3r, open3d, cv2).
- Each method's `can_run()` must be tested without loading the actual model.
- All model loads should be lazy (on first `reconstruct()` call) to keep startup fast.
- Reuse `_repair` / `_optimize` / `_scale_and_export` hooks in `ReconstructionEngine`.

---

## Pre-existing Issues (Not Addressed)

- `tests/test_mvp.py::MVPTests::test_scale_command_uses_input_dimensions` — `KeyError: 'scaleFactor'`
  (pre-existing, unrelated to reconstruction refactor)
- 47 tests skipped (PySide6/display not available in CI environment)

---

## Security Summary

CodeQL analysis: **0 alerts** — no security vulnerabilities introduced.
