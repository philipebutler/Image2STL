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
- [ ] Add installers for Windows/macOS packaging targets
- [ ] Run platform-specific acceptance testing on real image sets
- [ ] Load and render actual OBJ/STL model geometry in 3D preview control

## Suggested Next Execution Order

1. ~~UI shell + project workflow wiring~~ ✅
2. ~~Python inference backend substitution for mock pipeline~~ ✅
3. ~~Mesh repair and quality controls~~ ✅
4. ~~Export/preview UX refinement~~ ✅
5. 3D preview with actual model geometry (replace wireframe cube)
6. Packaging and acceptance testing
