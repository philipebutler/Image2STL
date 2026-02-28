# Phase 2 Handoff — Reconstruction Method Stubs

**Date:** 2026-02-28
**Phase:** 2 of 6
**Status:** ✅ Complete

---

## What Was Delivered

### New Files

```
core/reconstruction/
├── methods/
│   ├── __init__.py                  # Updated: exports all 4 method classes
│   ├── method_e_hybrid.py           # MethodEHybrid — checks GPU ≥6 GB + COLMAP
│   ├── method_d_dust3r.py           # MethodDDust3R — checks GPU ≥4 GB
│   ├── method_c_triposr.py          # MethodCTripoSR — always available (CPU-capable)
│   └── method_cloud.py              # MethodCloud — requires Meshy.ai API key in config
└── components/
    ├── __init__.py                  # Updated: exports all 4 component classes
    ├── view_synthesizer.py          # ViewSynthesizer stub (SyncDreamer/Zero123++)
    ├── colmap_wrapper.py            # COLMAPWrapper stub (SfM + dense)
    ├── mesh_aligner.py              # MeshAligner stub (ICP via trimesh/Open3D)
    └── mesh_verifier.py             # MeshVerifier stub (reprojection quality score)
```

Updated `core/reconstruction/__init__.py` to export all new classes.

### Key Classes

| Class | File | `can_run()` logic |
|---|---|---|
| `MethodEHybrid` | `methods/method_e_hybrid.py` | GPU + VRAM ≥ 6 GB + COLMAP installed |
| `MethodDDust3R` | `methods/method_d_dust3r.py` | GPU + VRAM ≥ 4 GB |
| `MethodCTripoSR` | `methods/method_c_triposr.py` | Always `True` (CPU-capable) |
| `MethodCloud` | `methods/method_cloud.py` | `config.get("meshyApiKey")` must be non-empty |
| `ViewSynthesizer` | `components/view_synthesizer.py` | `synthesize()` raises `NotImplementedError` |
| `COLMAPWrapper` | `components/colmap_wrapper.py` | `run_sfm()` / `run_dense()` raise `NotImplementedError` |
| `MeshAligner` | `components/mesh_aligner.py` | `align()` raises `NotImplementedError` |
| `MeshVerifier` | `components/mesh_verifier.py` | `score()` / `is_watertight()` raise `NotImplementedError` |

### Stub Contract (all four method classes)

- `can_run()` → real hardware checks via `MethodSelector.detect_hardware()`
- `estimate_time(n)` → method-specific formula using base from `get_method_requirements()`
- `reconstruct()` → returns `ReconstructionResult(success=False, error_message="Not yet implemented")`
- `get_method_name()` → unique human-readable string

### Tests

- **File:** `tests/test_reconstruction_methods.py`
- **Count:** 40 new tests, all passing (80 total including Phase 1)
- **Coverage:**
  - `MethodEHybrid.can_run()` — no GPU, insufficient VRAM, no COLMAP, ready (4 tests)
  - `MethodEHybrid` estimate_time + reconstruct stub + name (3 tests)
  - `MethodDDust3R.can_run()` — no GPU, insufficient VRAM, ready (3 tests)
  - `MethodDDust3R` estimate_time + reconstruct stub + name (3 tests)
  - `MethodCTripoSR.can_run()` — always true, no GPU still true (2 tests)
  - `MethodCTripoSR` estimate_time + reconstruct stub + name (3 tests)
  - `MethodCloud.can_run()` — no config, no API key, with API key (3 tests)
  - `MethodCloud` estimate_time constant + reconstruct stub + name (3 tests)
  - All 4 component stubs importable + raise `NotImplementedError` (10 tests)
  - Package-level import for all 8 new classes (2 tests)

---

## Design Decisions

1. **`can_run()` calls `MethodSelector.detect_hardware()` at call time** — avoids caching
   stale GPU state; negligible overhead since hardware rarely changes mid-session.
2. **`estimate_time()` reads base from `get_method_requirements()`** — single source of
   truth for method timing constants; no magic numbers in method files.
3. **`MethodCloud.can_run()` checks `config.get("meshyApiKey")`** — matches the existing
   `core/reconstruction_engine.py` pattern where `apiKey` is passed through the command dict.
4. **Component stubs raise `NotImplementedError` with descriptive messages** — immediately
   tells Phase 5 implementors exactly what each method does, without silent pass-through.

---

## What Comes Next (Phase 3)

### Objective
Build the `ReconstructionEngine` orchestrator that attempts methods in priority order
(E → D → C → Cloud) with automatic fallback on failure, post-processing pipeline,
and wires into the existing `core/reconstruction_engine.py` threading model.

### Files to Create / Modify
- `core/reconstruction/engine.py` — new `ReconstructionEngine` class with:
  - `reconstruct(images, output_dir, method_chain)` orchestration loop
  - `MethodAttempt` dataclass for attempt tracking
  - Post-processing pipeline: repair → optimize → scale → export
  - Progress aggregation across methods
- `core/reconstruction_engine.py` — wire new engine into existing thread wrapper

### Tests to Add
- Successful first-method path (mock method returns success)
- Fallback to second method when first fails
- All-methods-failed error reporting
- Post-processing pipeline invocation ordering

---

## Running Tests

```bash
# Phase 2 tests only
python -m pytest tests/test_reconstruction_methods.py -v

# Phase 1 + 2 tests
python -m pytest tests/test_reconstruction_foundation.py tests/test_reconstruction_methods.py -v

# Full suite (expect 1 pre-existing failure in test_mvp.py)
python -m pytest -q
```

---

## Pre-existing Issues (Not Addressed)

- `tests/test_mvp.py::MVPTests::test_scale_command_uses_input_dimensions` — `KeyError: 'scaleFactor'`
  (pre-existing, unrelated to reconstruction refactor)
- 19 tests skipped (optional dependencies not installed in CI environment)
