# Multi-Method 3D Reconstruction Refactor — Implementation Plan

This plan breaks the `big_refactor1.md` specification into appropriately sized slices
for iterative Copilot development. Each phase is self-contained, testable, and produces
a handoff file for the next session.

**Source specification:** `big_refactor1.md`

---

## Phase 1: Foundation & Infrastructure

**Goal:** Create the reconstruction subsystem skeleton with hardware detection, method
selection, and the abstract base class that all reconstruction methods will implement.

**Scope:**
- Create `core/reconstruction/` package structure (`__init__.py` for package, methods/, components/)
- Implement `ReconstructionMethod` enum and `HardwareCapabilities` dataclass
- Implement `MethodSelector` class with hardware detection, method selection, and fallback chain logic
- Implement `BaseReconstructor` abstract class with progress callback plumbing
- Implement `ReconstructionResult` dataclass for standardized method outputs
- Add unit tests covering:
  - `HardwareCapabilities` property logic (can_run_method_e/d/c)
  - `MethodSelector.select_method()` fallback chain ordering
  - `MethodSelector.get_method_requirements()` data
  - `BaseReconstructor.validate_inputs()` boundary checks
  - `ReconstructionResult` dataclass initialization

**Deliverables:** New package structure, tested foundation classes, handoff file.

**Estimated size:** ~5 new files, ~1 new test file, ~400 lines of production code.

---

## Phase 2: Reconstruction Method Stubs

**Goal:** Create concrete stub implementations of all three local reconstruction methods
(E, D, C) and the shared component interfaces so the fallback engine can be wired.

**Scope:**
- Implement `MethodEHybrid` stub (can_run checks, reconstruct returns NotImplemented-style result)
- Implement `MethodDDust3R` stub (same pattern)
- Implement `MethodCTripoSR` stub (same pattern)
- Adapt existing Meshy.ai cloud path to `BaseReconstructor` interface as `MethodCloud`
- Create shared component stubs: `ViewSynthesizer`, `COLMAPWrapper`, `MeshAligner`, `MeshVerifier`
- Add tests for each method's `can_run()` and `get_method_name()`

**Deliverables:** All method stubs inheriting from BaseReconstructor, component stubs.

**Estimated size:** ~8 new files, ~1 new test file, ~500 lines.

---

## Phase 3: Reconstruction Engine with Fallback

**Goal:** Build the orchestrator that attempts methods in priority order and falls back
automatically on failure.

**Scope:**
- Implement `ReconstructionEngine` (the new one in `core/reconstruction/`) with:
  - Method priority chain execution (E → D → C → Cloud)
  - Automatic fallback on method failure
  - Post-processing pipeline (repair → optimize → scale → export)
  - Progress aggregation across methods
  - Attempt tracking (`MethodAttempt` records)
- Wire the new engine into the existing `core/reconstruction_engine.py` thread wrapper
- Add tests for:
  - Successful first-method path
  - Fallback to second/third method
  - All-methods-failed error reporting
  - Post-processing pipeline invocation

**Deliverables:** Working fallback engine with integration to existing threading model.

**Estimated size:** ~2 new/modified files, ~1 test file, ~400 lines.

---

## Phase 4: UI Integration

**Goal:** Update the desktop UI to support method selection, hardware info display, and
multi-stage progress tracking.

**Scope:**
- Update `ui/widgets/control_panel.py` with method selection radio buttons (Auto/E/D/C/Cloud)
  - Grey out unavailable methods based on hardware detection
  - Show estimated time per method
- Add `ui/widgets/method_status_widget.py` showing which method is running and attempt history
- Add `ui/dialogs/hardware_info_dialog.py` showing detected hardware capabilities
- Update `ui/widgets/progress_widget.py` for multi-stage progress (method attempt + post-processing)
- Wire new signals: `method_started`, `method_completed`, `reconstruction_completed/failed`

**Deliverables:** Updated UI with method selection and hardware awareness.

**Estimated size:** ~3 new/modified files, ~400 lines.

---

## Phase 5: Full Method Implementations

**Goal:** Replace method stubs with real reconstruction logic using actual AI models.

**Scope:**
- **Method C (TripoSR Fusion):** Integrate TripoSR model loading, per-image reconstruction,
  ICP mesh alignment, and mesh fusion
- **Method D (Dust3R):** Integrate Dust3R model, pairwise reconstruction, global scene building,
  Poisson surface reconstruction
- **Method E (Hybrid Photogrammetry):** Integrate SyncDreamer view synthesis, COLMAP wrapper,
  geometry verification pipeline
- Implement shared components fully:
  - `ViewSynthesizer`: SyncDreamer/Zero123 integration
  - `COLMAPWrapper`: COLMAP subprocess management
  - `MeshAligner`: ICP alignment via trimesh/Open3D
  - `MeshVerifier`: Reprojection-based quality scoring

**Deliverables:** Working local reconstruction with all three methods.

**Estimated size:** ~10 modified files, ~2000 lines. Dependencies: torch, dust3r, open3d, cv2.

---

## Phase 6: Testing & Polish

**Goal:** End-to-end testing, error handling edge cases, and performance tuning.

**Scope:**
- Integration tests with mock/lightweight models for full fallback chain
- Platform-specific hardware detection tests
- UI smoke tests for method selection workflows
- Error handling: GPU OOM recovery, COLMAP timeout, model download failures
- Performance: memory profiling, GPU memory cleanup between method attempts
- Update `requirements.txt` with all new dependencies
- Update `SPEC.md` and `README.md` documentation

**Deliverables:** Production-ready multi-method reconstruction system.

**Estimated size:** ~3 test files, ~500 lines. Documentation updates.

---

## Dependency Graph

```
Phase 1 (Foundation)
    ↓
Phase 2 (Method Stubs)
    ↓
Phase 3 (Fallback Engine)
    ↓
Phase 4 (UI Integration)    Phase 5 (Full Methods)
    ↓                            ↓
         Phase 6 (Testing & Polish)
```

Phases 4 and 5 can be executed in parallel once Phase 3 is complete.

---

## Risk Notes

| Risk | Mitigation |
|------|------------|
| GPU not available in CI | All foundation/stub tests run CPU-only; mock hardware capabilities |
| Large model downloads | Phase 5 uses lazy loading; tests use mocks |
| COLMAP binary dependency | Method E gracefully degrades; `check_colmap_installed()` tested |
| PySide6 not available in headless CI | UI tests skipped when display unavailable |
