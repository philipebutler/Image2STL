# 3D Head Scanner MVP Specification

## Project Overview

A desktop application for creating 3D-printable models from smartphone photos. Primary use case is scanning people's heads, but supports general object capture. The application uses a Python UI frontend and Python-based AI reconstruction backend.

## Target Users

- Hobbyists creating personalized 3D prints
- Makers capturing physical objects for modification
- Users with typical consumer laptops (no GPU required)

## Core Requirements

### Functional Requirements

#### FR1: Project Management
- Create new projects with unique identifiers
- Save project state to disk (images, settings, generated models)
- Load existing projects to resume work
- Project file format: JSON with metadata + associated files in project folder

#### FR2: Image Input
- Accept 3-5 smartphone images per project
- Support common formats: JPG, PNG, HEIC
- Input methods:
  - Drag and drop images onto application window
  - File picker dialog for manual selection
- Display thumbnails of loaded images in UI

#### FR3: 3D Reconstruction
- **Multi-Method Engine**: Attempts methods in priority order (E → D → C → Cloud) with automatic fallback
  - **Method E (Hybrid Photogrammetry)**: Combines real photos with AI-generated views (SyncDreamer/Zero123++) and COLMAP. Requires 6+ GB VRAM and COLMAP. Highest quality.
  - **Method D (Dust3R Multi-View)**: Pairwise AI reconstruction using Dust3R. Requires 4+ GB VRAM. High quality.
  - **Method C (TripoSR Fusion)**: Per-image TripoSR inference + ICP alignment + mesh fusion. CPU-capable. Good quality.
  - **Cloud (Meshy.ai)**: Cloud-based reconstruction via Meshy.ai API. Requires API key and internet.
- Hardware detection selects the best available method automatically
- User can override with manual method selection (Auto / E / D / C / Cloud)
- Unavailable methods are greyed out in the UI based on detected hardware
- Estimated processing time per method is displayed
- Target processing time: 3-10 minutes depending on method
- Warn user if estimated time >10 minutes

#### FR4: Mesh Processing
- Automatic mesh repair to ensure 3D printability:
  - Watertight (no holes)
  - Manifold geometry (printable topology)
  - Remove internal geometry
  - Fix normals
- Use PyMeshLab or similar for repair operations

#### FR5: 3D Preview
- Interactive 3D viewer showing reconstructed model
- Controls:
  - Rotate (mouse drag)
  - Zoom (scroll wheel)
- Display before export to verify quality

#### FR6: Scaling
- Allow user to specify dimensions in millimeters
- Apply to longest axis, width, height, or depth
- Default to reasonable print size (e.g., 150mm for heads)
- Preview shows current dimensions

#### FR7: Export
- Export final model as STL file
- User specifies output location via save dialog
- Include project name in default filename

#### FR8: Progress Feedback
- Progress bar during reconstruction process
- Status messages for key steps:
  - "Loading images..."
  - "Running AI reconstruction..."
  - "Repairing mesh..."
  - "Generating preview..."
- Cancel operation button (best effort)

#### FR9: Error Handling
- Detect and report specific errors:
  - Insufficient images
  - Image quality too poor
  - Reconstruction failed
  - API errors (cloud mode)
  - Mesh repair failures
- User-friendly error messages with suggested actions

### Non-Functional Requirements

#### NFR1: Platform Support
- Windows 10/11 (64-bit)
- macOS 11+ (Intel and Apple Silicon)

#### NFR2: Performance
- Reconstruction time: 5-10 minutes target
- UI remains responsive during processing
- Memory usage: <8GB RAM

#### NFR3: Installation
- Single installer package per platform
- Windows: MSI or NSIS installer
- macOS: DMG with app bundle
- Installer includes:
  - Python UI application
  - Bundled Python reconstruction engine
  - Pre-trained AI models
- Installation size: ~2-3GB

#### NFR4: Usability
- Simple, clean interface
- No technical knowledge required
- Clear visual feedback at all stages

## Architecture

### Component Overview

```
┌─────────────────────────────────────┐
│      Python UI Application           │
│  ┌──────────────────────────────┐   │
│  │  Project Management          │   │
│  │  Image Loading/Preview       │   │
│  │  3D Viewer (Python/OpenGL)   │   │
│  │  Method Selection / HW Info  │   │
│  │  Settings & Controls         │   │
│  └──────────────────────────────┘   │
└──────────────┬──────────────────────┘
               │ in-process calls
               │
┌──────────────▼──────────────────────┐
│   Multi-Method Reconstruction Engine │
│  ┌──────────────────────────────┐   │
│  │  HardwareCapabilities detect │   │
│  │  MethodSelector (E→D→C→Cloud)│   │
│  │  ReconstructionEngine        │   │
│  │  ├─ MethodEHybrid            │   │
│  │  │   ├─ ViewSynthesizer      │   │
│  │  │   ├─ COLMAPWrapper        │   │
│  │  │   └─ MeshVerifier         │   │
│  │  ├─ MethodDDust3R            │   │
│  │  ├─ MethodCTripoSR           │   │
│  │  │   └─ MeshAligner (ICP)    │   │
│  │  └─ MethodCloud (Meshy.ai)   │   │
│  │  Post-processing pipeline:   │   │
│  │   repair → optimize → scale  │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

### Technology Stack

#### Frontend (Python UI)
- **Framework**: PySide6
- **3D Viewer**: OpenGL-based wireframe preview
- **JSON**: Python stdlib json
- **Project Structure**: modular UI components

#### Backend (Python)
- **Python**: 3.10+
- **Multi-Method AI Engine**: MethodEHybrid, MethodDDust3R, MethodCTripoSR, MethodCloud
- **View Synthesis**: Zero123++ (diffusers) or SyncDreamer
- **Photogrammetry**: COLMAP (external binary, Method E only)
- **Mesh Processing**: trimesh (repair, ICP alignment, decimation, scaling)
- **Cloud API**: Meshy.ai Python SDK
- **Dependencies**: PyTorch, numpy, Pillow, trimesh, open3d, opencv-python, diffusers
- **Packaging**: PyInstaller with --onefile

#### Installer
- **Windows**: WiX Toolset or NSIS
- **macOS**: create-dmg or appdmg

## Data Flow

### Project Creation Flow
1. User clicks "New Project"
2. App creates project folder structure
3. User adds 3-5 images via drag-drop or file picker
4. App copies images to project folder
5. App saves project.json with metadata

### Reconstruction Flow
1. User selects Local or Cloud mode
2. User clicks "Generate 3D Model"
3. Frontend validates inputs (3-5 images)
4. Frontend launches Python engine subprocess
5. Frontend sends JSON message: `{"command": "reconstruct", "mode": "local|cloud", "images": [...], "project_id": "..."}`
6. Python engine:
   - Loads images
   - Runs reconstruction (TripoSR or Meshy.ai)
   - Repairs mesh
   - Saves intermediate .obj/.stl
   - Sends progress updates via stdout: `{"progress": 0.5, "status": "Running AI reconstruction..."}`
7. Python engine sends completion: `{"status": "complete", "model_path": "..."}`
8. Frontend loads model into 3D viewer

### Export Flow
1. User adjusts scale in millimeters
2. User clicks "Export STL"
3. Frontend sends scale command to Python engine (or applies in UI layer if mesh already loaded)
4. User selects save location
5. App exports STL file
6. App shows success message

## File Structure

### Application Installation
```
HeadScanner/
├── HeadScanner.exe (or .app on macOS)
├── reconstruction-engine/
│   └── reconstruction.exe (bundled Python)
└── models/
    └── triposr-model/ (pre-trained weights)
```

### Project Folder Structure
```
MyProject/
├── project.json
├── images/
│   ├── IMG_001.jpg
│   ├── IMG_002.jpg
│   └── ...
├── models/
│   ├── raw_reconstruction.obj
│   ├── repaired_mesh.obj
│   └── final_export.stl
└── preview/
    └── thumbnail.png
```

### project.json Schema
```json
{
  "projectId": "unique-guid",
  "name": "MyHeadScan",
  "created": "2026-02-21T10:30:00Z",
  "lastModified": "2026-02-21T11:45:00Z",
  "images": [
    "images/IMG_001.jpg",
    "images/IMG_002.jpg"
  ],
  "reconstructionMode": "local",
  "modelPath": "models/final_export.stl",
  "scaleMm": 150.0,
  "settings": {
    "meshQuality": "medium"
  }
}
```

## Python Engine API

### Communication Protocol
- Transport: stdin/stdout
- Format: Line-delimited JSON

### Commands (Frontend → Engine)

#### Reconstruct
```json
{
  "command": "reconstruct",
  "mode": "local",
  "images": ["/path/to/img1.jpg", "/path/to/img2.jpg"],
  "outputPath": "/path/to/output.obj",
  "projectId": "abc-123"
}
```

#### Repair Mesh
```json
{
  "command": "repair",
  "inputMesh": "/path/to/raw.obj",
  "outputMesh": "/path/to/repaired.stl"
}
```

#### Scale Mesh
```json
{
  "command": "scale",
  "inputMesh": "/path/to/mesh.stl",
  "outputMesh": "/path/to/scaled.stl",
  "targetSizeMm": 150.0,
  "axis": "longest"
}
```

### Responses (Engine → Frontend)

#### Progress Update
```json
{
  "type": "progress",
  "progress": 0.35,
  "status": "Running AI reconstruction...",
  "estimatedSecondsRemaining": 180
}
```

#### Success
```json
{
  "type": "success",
  "command": "reconstruct",
  "outputPath": "/path/to/model.obj",
  "stats": {
    "vertices": 50000,
    "faces": 100000
  }
}
```

#### Error
```json
{
  "type": "error",
  "command": "reconstruct",
  "errorCode": "INSUFFICIENT_FEATURES",
  "message": "Could not detect enough features in images. Try taking photos from more angles.",
  "suggestion": "Add 2-3 more images with different viewpoints"
}
```

## UI Mockup (Text Description)

### Main Window Layout

```
┌────────────────────────────────────────────────┐
│ File  Project  Help                            │
├────────────────────────────────────────────────┤
│                                                │
│  ┌──────────────┐  ┌─────────────────────────┐│
│  │              │  │                         ││
│  │   Image      │  │                         ││
│  │   Gallery    │  │    3D Preview           ││
│  │              │  │    (rotate/zoom)        ││
│  │  [img] [img] │  │                         ││
│  │  [img] [img] │  │                         ││
│  │              │  │                         ││
│  │  [Add More]  │  │                         ││
│  │              │  └─────────────────────────┘│
│  └──────────────┘                             │
│                                                │
│  Reconstruction Mode: ○ Local  ○ Cloud        │
│                                                │
│  Scale (mm): [150] (longest axis)             │
│                                                │
│  [Generate 3D Model]  [Export STL]            │
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │ Progress: ████████░░░░░░░░ 45%           │ │
│  │ Status: Running AI reconstruction...     │ │
│  └──────────────────────────────────────────┘ │
└────────────────────────────────────────────────┘
```

## Error Codes

| Code | Meaning | User Message | Suggestion |
|------|---------|--------------|------------|
| INSUFFICIENT_IMAGES | <3 images | "Not enough images" | "Add at least 3 images" |
| TOO_MANY_IMAGES | >5 images | "Too many images for MVP" | "Use 3-5 images" |
| INSUFFICIENT_FEATURES | Feature detection failed | "Images too similar or low quality" | "Try different angles or better lighting" |
| RECONSTRUCTION_FAILED | AI model error | "3D reconstruction failed" | "Try different images or cloud mode" |
| API_ERROR | Cloud service error | "Cloud service unavailable" | "Try local mode or retry later" |
| MESH_REPAIR_FAILED | Mesh not repairable | "Could not create printable mesh" | "Try different source images" |
| FILE_IO_ERROR | Disk operation failed | "Could not save/load file" | "Check disk space and permissions" |

## Development Phases

### Phase 1: Core Infrastructure (Week 1-2)
- Set up Python desktop UI project structure
- Implement basic project management (create, save, load)
- Create image loading UI (drag-drop, file picker)
- Set up Python engine skeleton with IPC

### Phase 2: Reconstruction Pipeline (Week 3-4)
- Integrate TripoSR for local reconstruction
- Implement Meshy.ai cloud integration
- Add mesh repair using PyMeshLab
- Build progress reporting system

### Phase 3: Visualization & Export (Week 5)
- Integrate 3D viewer component
- Implement rotation/zoom controls
- Add scaling functionality
- Implement STL export

### Phase 4: Polish & Packaging (Week 6)
- Error handling and user messages
- Processing time warnings
- Build PyInstaller bundling
- Create installers for Windows/macOS

### Phase 5: Testing & Refinement (Week 7)
- End-to-end testing with real photos
- Performance optimization
- Bug fixes
- Documentation

## Testing Requirements

### Unit Tests
- Project serialization/deserialization
- Image validation
- Scale calculations
- IPC message parsing

### Integration Tests
- Full reconstruction pipeline (local mode)
- Full reconstruction pipeline (cloud mode)
- Mesh repair workflow
- Export workflow

### User Acceptance Tests
- Create project with head photos → export printable STL
- Save and load project successfully
- Handle errors gracefully
- Process completes in <10 minutes

## Dependencies

### Python UI Packages
- PySide6
- pyqtgraph or OpenGL helper package (for 3D preview)

### Python Packages
- torch (CPU-only build)
- transformers
- trimesh
- pymeshlab
- pillow
- numpy
- requests (for Meshy.ai)
- huggingface-hub

### External Services
- Meshy.ai API (requires API key, user-provided or embedded)

## Configuration

### appsettings.json (Desktop UI)
```json
{
  "PythonEngine": {
    "ExecutablePath": "reconstruction-engine/reconstruction.exe",
    "TimeoutSeconds": 900,
    "MaxMemoryMB": 8192
  },
  "Defaults": {
    "ScaleMm": 150.0,
    "ReconstructionMode": "local"
  },
  "MeshyApi": {
    "BaseUrl": "https://api.meshy.ai/v1",
    "TimeoutSeconds": 600
  }
}
```

### config.json (Python Engine)
```json
{
  "triposr": {
    "modelPath": "../models/triposr-model",
    "device": "cpu",
    "chunkSize": 8192
  },
  "meshRepair": {
    "targetFaceCount": 100000,
    "fillHoles": true,
    "removeInternalGeometry": true
  }
}
```

## Security Considerations

- API keys: Store Meshy.ai API key securely (encrypted local storage or user-provided)
- File paths: Validate all user-provided paths to prevent directory traversal
- Subprocess: Sanitize inputs passed to Python engine
- Project files: Validate JSON before parsing to prevent injection

## Future Enhancements (Post-MVP)

- Batch processing multiple projects
- Texture capture and UV mapping
- Advanced mesh editing tools
- Direct printer integration
- Mobile app for image capture
- Quality presets (draft/standard/high)
- Multi-angle shooting guide
- Cloud storage for projects
- Community model sharing

## Success Metrics

- Successfully reconstruct and export printable head model in <10 minutes
- 90% of reconstructions pass mesh validation
- Application runs on typical consumer hardware (8GB RAM, no GPU)
- Installer size <3GB
- User can complete full workflow without documentation

## Open Questions for Development

1. 3D viewer compatibility in the Python UI - may need alternative rendering backend
   - **Resolution**: Build 4-hour spike in Week 1 to validate
2. TripoSR PyInstaller bundling complexity - may need custom build process
   - **Resolution**: Create bundling test in Week 1, adjust if bundle >5GB
3. Meshy.ai API rate limits and costs - determine pricing tier needed
   - **Resolution**: Research and sign up in Week 2 before Phase 2

## MVP Exclusions

- ❌ Code signing certificates (users will see security warnings, document bypass instructions)
- ❌ Auto-update mechanism (defer to post-MVP)
- ❌ App notarization for macOS (users must right-click > Open on first launch)

---

## Phase 0: Technical Validation (Week 1)

Before starting full development, validate critical technical assumptions through focused proof-of-concept implementations.

### Validation 1: 3D Viewer Compatibility

**Goal**: Confirm the selected Python UI 3D viewer works for STL display with rotate/zoom controls.

**Tasks**:
1. Create new Python desktop UI test application
2. Add/validate selected 3D viewer dependency
3. Create a minimal window with 3D viewport
4. Load a sample STL file (cube or sphere)
5. Implement basic mouse controls:
   - Left-click drag = rotate
   - Scroll wheel = zoom
6. Test on both Windows and macOS

**Success Criteria**:
- ✅ STL file loads and displays correctly
- ✅ Smooth rotation with mouse drag
- ✅ Zoom works with scroll wheel
- ✅ Renders at acceptable framerate (>30 fps)
- ✅ Works on both Windows and macOS

**Failure Scenarios & Alternatives**:

If default 3D viewer doesn't work:
- **Option A**: Try alternate Python 3D rendering packages
- **Option B**: Embed web-based viewer (Babylon.js/Three.js) in a web view
- **Option C**: Use a custom OpenGL renderer
- **Option D**: Shell out to external viewer app (Meshlab, etc.) - least desirable

**Time Budget**: 4-6 hours

**Deliverable**: Working sample app that displays and rotates an STL file, or documented decision on alternative approach.

---

### Validation 2: Python Engine Bundling

**Goal**: Confirm TripoSR can be bundled into a single executable with acceptable size and performance.

**Tasks**:

1. **Set up Python environment**:
   ```bash
   python -m venv triposr-test
   source triposr-test/bin/activate  # or triposr-test\Scripts\activate on Windows
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install transformers trimesh pymeshlab pillow
   ```

2. **Create test script** (`test_triposr.py`):
   ```python
   import sys
   import json
   from transformers import pipeline
   import trimesh
   
   def main():
       # Simple test: load TripoSR and generate from single image
       print(json.dumps({"status": "loading_model"}))
       # Load model code here
       print(json.dumps({"status": "ready"}))
       
       # Accept JSON command from stdin
       for line in sys.stdin:
           cmd = json.loads(line)
           if cmd.get("command") == "test":
               print(json.dumps({"status": "success", "message": "Engine works"}))
   
   if __name__ == "__main__":
       main()
   ```

3. **Test PyInstaller bundling**:
   ```bash
   pip install pyinstaller
   
   # Create spec file with hidden imports
   pyinstaller --onefile \
     --name reconstruction-engine \
     --hidden-import=torch \
     --hidden-import=transformers \
     --hidden-import=trimesh \
     --hidden-import=pymeshlab \
     --collect-all torch \
     --collect-all transformers \
     test_triposr.py
   ```

4. **Measure results**:
   - Executable size
   - Startup time
   - Memory usage during inference
   - Test on clean VM without Python installed

5. **Test actual TripoSR inference**:
   ```python
   from tsr.system import TSR
   from PIL import Image
   
   model = TSR.from_pretrained("stabilityai/TripoSR")
   image = Image.open("test_image.jpg")
   mesh = model.run_image(image)
   mesh.export("output.obj")
   ```

**Success Criteria**:
- ✅ Bundle completes without errors
- ✅ Executable size <3GB (ideally <2GB)
- ✅ Runs on clean Windows/macOS without Python installed
- ✅ Can perform inference on test image
- ✅ Startup time <30 seconds
- ✅ Peak memory usage <8GB

**Failure Scenarios & Alternatives**:

If bundling fails or is too large:
- **Option A**: Don't bundle PyTorch models, download on first run
- **Option B**: Use ONNX runtime instead of PyTorch (smaller, faster)
- **Option C**: Switch to lighter model (InstantMesh, Wonder3D-lite)
- **Option D**: Require separate Python installation (document setup steps)
- **Option E**: Cloud-only mode for MVP, add local later

**Time Budget**: 8-12 hours (includes model testing)

**Deliverable**: 
- Working bundled executable OR
- Documented alternative approach with revised architecture
- Actual size/performance metrics

---

### Validation 3: Meshy.ai API Integration

**Goal**: Understand Meshy.ai pricing, rate limits, and API capabilities for the MVP.

**Tasks**:

1. **Research and documentation**:
   - Visit https://docs.meshy.ai
   - Review pricing at https://www.meshy.ai/pricing
   - Document:
     - Cost per text-to-3D or image-to-3D generation
     - Rate limits (requests per minute/hour/day)
     - Free tier limitations
     - Expected wait time for generation
     - Output formats available (need .obj or .stl)

2. **Sign up and test**:
   ```bash
   pip install meshy-ai  # or requests if no SDK
   ```

3. **Create test integration** (`test_meshy.py`):
   ```python
   import os
   from meshy import MeshyClient  # Adjust based on actual SDK
   
   client = MeshyClient(api_key=os.environ["MESHY_API_KEY"])
   
   # Test image-to-3D
   task = client.create_image_to_3d(
       image_path="test_head.jpg",
       topology="quad"  # or whatever options they offer
   )
   
   # Poll for completion
   while task.status != "completed":
       time.sleep(5)
       task = client.get_task(task.id)
   
   # Download result
   task.download("output.obj")
   ```

4. **Measure and document**:
   - Actual cost per generation
   - Time to complete (compare to 5-10 min target)
   - Quality of output mesh
   - Whether mesh repair is still needed
   - API error handling

5. **Cost analysis for MVP**:
   - Estimate: If testing with 20 projects = $X
   - Estimate: If 100 users each do 5 projects = $Y
   - Determine if costs are acceptable

**Success Criteria**:
- ✅ API key obtained and working
- ✅ Can successfully generate 3D model from images
- ✅ Generation completes in <15 minutes
- ✅ Output format is usable (.obj, .stl, or .glb)
- ✅ Cost per generation is acceptable for MVP budget
- ✅ Rate limits allow reasonable testing workflow

**Failure Scenarios & Alternatives**:

If Meshy.ai doesn't work:
- **Option A**: Try alternative services:
  - Luma AI (https://lumalabs.ai)
  - Kaedim (https://www.kaedim3d.com)
  - Polycam API
- **Option B**: Local-only for MVP, add cloud later
- **Option C**: Use free trial credits only, require users to provide API key

**Time Budget**: 4-6 hours

**Deliverable**:
- Working API integration sample OR
- Decision on alternative cloud service
- Cost/limits documentation for planning

---

## Validation Summary Table

| Validation | Time | Risk Level | Blocking? | Alternatives Available? |
|------------|------|------------|-----------|------------------------|
| 3D Viewer | 4-6h | Medium | Yes | Multiple good options |
| Python Bundling | 8-12h | High | Yes | Can pivot architecture |
| Meshy.ai API | 4-6h | Low | No | Many alternative services |

**Total validation time**: 16-24 hours (2-3 days)

**Go/No-Go Decision Point**: End of Week 1
- If all 3 validations pass: Proceed with full development
- If 1-2 fail: Adjust architecture, re-validate alternatives
- If all 3 fail: Re-evaluate fundamental approach

---

## Instructions for Implementation

### For GitHub Copilot / AI Assistant:

When implementing Phase 0 validations:

1. **Create separate test projects** for each validation (don't pollute main codebase)
2. **Document all findings** in markdown files in `/validation-results/` directory
3. **Include actual metrics**: file sizes, timing measurements, memory usage
4. **Screenshot or record** 3D viewer working (visual proof)
5. **Save all error messages** if something fails
6. **Test on both platforms** (Windows and macOS) where applicable
7. **Create decision document** at end of Phase 0 recommending proceed/pivot/alternative

### Validation Checklist Template:

```markdown
# Validation: [Name]
Date: YYYY-MM-DD
Tester: [Name]

## Results
- [ ] Success Criterion 1
- [ ] Success Criterion 2
- [ ] ...

## Metrics
- Executable size: X MB
- Startup time: X seconds
- Memory usage: X MB
- ...

## Issues Encountered
1. Issue description
2. Workaround/solution

## Recommendation
☐ PASS - Proceed with this approach
☐ FAIL - Use Alternative: [name]
☐ NEEDS_WORK - [what needs fixing]

## Next Steps
- Action item 1
- Action item 2
```

### Repository Structure for Validation:

```
head-scanner/
├── validation/
│   ├── 01-3d-viewer/
│   │   ├── DesktopViewerTest/
│   │   ├── README.md
│   │   └── results.md
│   ├── 02-python-bundling/
│   │   ├── test_triposr.py
│   │   ├── build.sh
│   │   ├── README.md
│   │   └── results.md
│   └── 03-meshy-api/
│       ├── test_meshy.py
│       ├── README.md
│       └── results.md
└── docs/
    └── validation-summary.md
```

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-21  
**Author**: Development Team  
**Status**: Ready for Implementation
