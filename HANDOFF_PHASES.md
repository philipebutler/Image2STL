# MVP Follow-on Phase Handoff

The implemented MVP delivers core command and data-model behavior from `SPEC.md` with focused tests.

## Remaining Phase Work

- [x] Wire project workflow shell (create/load/add-images/reconstruct) in CLI
- [x] Build full desktop UI (image gallery thumbnails, drag-drop, file picker)
- [x] Integrate interactive 3D viewer with rotate/zoom
- [x] Replace mock reconstruction with TripoSR local inference
- [x] Integrate Meshy.ai API client for cloud mode with API key management
- [x] Implement robust mesh repair pipeline using PyMeshLab/trimesh
- [x] Add cancel-operation plumbing and long-running process management
- [x] Add Desktop UI controls: mode toggle, scale inputs, progress bar, export STL, keyboard shortcuts
- [x] Add image quality pre-flight checks and WebP/AVIF format support
- [x] Load and render actual OBJ/STL model geometry in 3D preview control
- [x] Add assumption-guided reconstruction controls (flat-bottom/symmetry/confidence)
- [x] Add assumption policy presets (`conservative`/`standard`/`aggressive`) wired across UI/core/CLI
- [x] Add user-facing assumption result summaries (status/progress info) including preset/threshold transparency
- [ ] Add installers for Windows/macOS packaging targets
- [ ] Run platform-specific acceptance testing on real image sets

## Suggested Next Execution Order

1. ~~UI shell + project workflow wiring~~ ✅
2. ~~Python inference backend substitution for mock pipeline~~ ✅
3. ~~Mesh repair and quality controls~~ ✅
4. ~~Export/preview UX refinement~~ ✅
5. ~~3D preview with actual model geometry (replace wireframe cube)~~ ✅
6. ~~Assumption-guided reconstruction hardening + presets~~ ✅
7. Packaging and acceptance testing

## Latest Slice Notes

- Reconstruction assumptions now support policy presets with conservative safety gating.
- Effective confidence threshold is computed as `max(user_threshold, preset_minimum)` and surfaced in UI summaries.
- CLI supports `--assumption-preset` and forwards to backend reconstruction commands.
- Preprocess command import path is hardened for tests that stub only `preprocess_image`.
- Cross-platform path assertions in tests were normalized (`resolve()`) to avoid macOS `/var` vs `/private/var` mismatches.
