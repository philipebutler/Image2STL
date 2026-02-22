# Recommended Changes and Updates

This document outlines recommended changes and updates to the Image2STL application to improve usability, reliability, and completeness relative to the SPEC.md requirements.

## High Priority

### 1. Native HEIC Thumbnail Rendering in Desktop UI

**Current State**: HEIC/HEIF files are accepted in the Desktop UI but display a blank placeholder thumbnail because Avalonia cannot natively decode HEIC images.

**Recommendation**: Use the `pillow-heif` Python backend (or a .NET HEIC library such as `LibHeifSharp`) to convert HEIC images to JPEG/PNG on import so that real thumbnails are displayed in the image gallery.

---

### 2. End-to-End HEIC Image Processing Verification

**Current State**: The Python engine registers `pillow-heif` for Pillow to open HEIC files, but the full TripoSR inference pipeline has not been validated with HEIC inputs on all platforms.

**Recommendation**: Add integration tests that feed actual HEIC sample images through the full reconstruction pipeline (local and cloud modes) and confirm the output mesh is valid.

---

### 3. Robust Mesh Repair Pipeline

**Current State**: The mesh repair command has a dual-path fallback (PyMeshLab → trimesh) but the pipeline is basic. Missing operations include decimation to a target face count and removal of internal geometry.

**Recommendation**: Implement the full mesh repair sequence described in SPEC.md (FR4), including configurable target face count, internal geometry removal, and automated watertight validation with clear pass/fail feedback to the user.

---

### 4. Cancel Operation Plumbing

**Current State**: Cancel support is implemented at the engine level via operation IDs, but there is no UI button or frontend-to-backend cancel message wiring in the Avalonia desktop application.

**Recommendation**: Add a "Cancel" button to the progress bar area of the main window. Wire it to send a `{"command": "cancel", "targetOperationId": "..."}` JSON message to the Python subprocess.

---

### 5. STL Export Workflow

**Current State**: The `scale` command writes a scaled file, but there is no explicit STL export button or save dialog in the Desktop UI. The SPEC (FR7) calls for a user-facing "Export STL" button with a save dialog that defaults to the project name.

**Recommendation**: Add an "Export STL" button in the main window that triggers a save-file dialog (defaulting to `<ProjectName>.stl`), then invokes the scale and export pipeline through the Python engine.

---

## Medium Priority

### 6. Processing Time Warning

**Current State**: The engine includes `estimatedSecondsRemaining` and a warning flag when estimated time exceeds 10 minutes, but the Desktop UI does not display this warning.

**Recommendation**: Display a yellow warning banner or dialog in the UI when `estimatedSecondsRemaining` exceeds 600 seconds, as specified in SPEC FR3 and NFR2.

---

### 7. 3D Preview with Actual Model Geometry

**Current State**: The 3D preview control (`WireframeViewerControl`) renders a hardcoded wireframe cube. It does not load or display the reconstructed mesh.

**Recommendation**: Replace or extend the wireframe control to load and render the actual OBJ/STL model output from reconstruction. Consider using a library such as `Silk.NET.OpenGL` or an embedded web viewer with Three.js. At minimum, parse the OBJ vertices and faces and render them as a wireframe.

---

### 8. Image Quality Feedback Before Reconstruction

**Current State**: Image quality problems (blur, low resolution, insufficient angles) are only detected after reconstruction fails with `INSUFFICIENT_FEATURES`.

**Recommendation**: Add pre-flight image quality checks before starting reconstruction. Detect and warn about images that are too small (e.g., below 512×512), blurry (Laplacian variance), or too similar to each other (feature matching).

---

### 9. Drag-and-Drop Visual Feedback

**Current State**: Drag-and-drop is functional but there is no visual feedback (highlight, overlay, or drop zone indicator) when files are dragged over the window.

**Recommendation**: Add a visual drop zone overlay that appears during drag-over to guide users. Change the window border or background to indicate that files can be dropped.

---

### 10. Error Recovery Guidance in UI

**Current State**: Error codes and suggestions are defined in the Python backend but the Desktop UI does not display user-friendly error messages or suggestions from the engine.

**Recommendation**: Parse the `message` and `suggestion` fields from engine error responses and display them in a dialog or status panel. This aligns with SPEC FR9.

---

## Lower Priority

### 11. Installer Packaging

**Current State**: Build scripts exist for Windows (PowerShell) and macOS (shell script) but complete installer packaging with bundled Python engine and pre-trained models has not been verified.

**Recommendation**: Complete and test the installer pipelines on both platforms. Include the Python engine (via PyInstaller), required models, and the .NET Avalonia application in a single installer package per SPEC NFR3.

---

### 12. Additional Image Format Support

**Current State**: Supported formats are JPG, PNG, HEIC, and HEIF. Other common smartphone formats like WebP and AVIF are not supported.

**Recommendation**: Consider adding WebP support (Pillow supports it natively) and AVIF support (via `pillow-avif-plugin`). These formats are increasingly common on Android and iOS devices.

---

### 13. Reconstruction Mode Toggle in UI

**Current State**: The SPEC (FR3) calls for a UI toggle/dropdown to select local vs. cloud mode before processing. The ViewModel does not expose a reconstruction mode property or corresponding UI element.

**Recommendation**: Add a radio button or dropdown in the main window for selecting "Local" or "Cloud" reconstruction mode, bound to the project's `reconstructionMode` property.

---

### 14. Scale Controls in UI

**Current State**: The SPEC (FR6) calls for user-adjustable scale in millimeters with axis selection. The Desktop UI does not expose these controls.

**Recommendation**: Add a numeric input for target size in millimeters and a dropdown for axis selection (longest/width/height/depth) in the main window, with a live display of current dimensions.

---

### 15. Keyboard Accessibility and Shortcuts

**Current State**: There are no keyboard shortcuts for common operations (New Project, Add Images, Generate, Export).

**Recommendation**: Add keyboard shortcuts such as Ctrl+N (New), Ctrl+O (Open), Ctrl+S (Save), Ctrl+I (Add Images), and Ctrl+G (Generate). This improves usability for power users and accessibility.

---

### 16. Application Settings and Preferences

**Current State**: Configuration values like default scale (150mm), mesh quality, and Python engine path are hardcoded.

**Recommendation**: Add a settings/preferences dialog or configuration file (`appsettings.json`) that allows users to customize defaults such as scale, reconstruction mode, mesh quality preset, and Meshy.ai API key storage, as described in the SPEC Configuration section.

---

### 17. Progress Bar Integration

**Current State**: Progress messages are generated by the engine but there is no progress bar widget in the Desktop UI to display them.

**Recommendation**: Add a progress bar and status label to the main window that updates in real time based on `progress` and `status` fields from engine messages during reconstruction.

---

### 18. Automated Test Coverage Expansion

**Current State**: Unit tests cover serialization, validation, commands, and CLI workflows. Integration tests with actual model inference and platform-specific UI tests are missing.

**Recommendation**: Add integration tests for the full reconstruction pipeline with sample images (using mocked or lightweight models), platform-specific UI smoke tests, and end-to-end tests for the export workflow.

---

## Summary

| Priority | Items | Key Theme |
|----------|-------|-----------|
| High | 1-5 | Core workflow completeness and HEIC reliability |
| Medium | 6-10 | User experience and feedback |
| Lower | 11-18 | Packaging, accessibility, and test coverage |
