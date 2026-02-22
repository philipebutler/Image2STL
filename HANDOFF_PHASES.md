# MVP Follow-on Phase Handoff

The implemented MVP delivers core command and data-model behavior from `SPEC.md` with focused tests.

## Remaining Phase Work

- [x] Wire project workflow shell (create/load/add-images/reconstruct) in CLI
- [ ] Build full Avalonia desktop UI (image gallery thumbnails, drag-drop, file picker)
- [ ] Integrate interactive 3D viewer with rotate/zoom
- [ ] Replace mock reconstruction with TripoSR local inference
- [ ] Integrate Meshy.ai API client for cloud mode with API key management
- [ ] Implement robust mesh repair pipeline using PyMeshLab/trimesh
- [ ] Add cancel-operation plumbing and long-running process management
- [ ] Add installers for Windows/macOS packaging targets
- [ ] Run platform-specific acceptance testing on real image sets

## Suggested Next Execution Order

1. UI shell + project workflow wiring
2. Python inference backend substitution for mock pipeline
3. Mesh repair and quality controls
4. Export/preview UX refinement
5. Packaging and acceptance testing
