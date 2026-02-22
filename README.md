# Image2STL MVP

Image2STL converts 3-5 smartphone images into a printable STL workflow with:

- local reconstruction (TripoSR)
- cloud reconstruction (Meshy.ai)
- mesh repair/scaling commands
- Avalonia desktop image gallery + 3D preview shell

Implementation scope and behavior are defined in `SPEC.md`.

## Install

### Python CLI dependencies

```bash
python -m pip install pillow torch transformers trimesh pymeshlab numpy
```

Optional format support:

```bash
python -m pip install pillow-heif    # HEIC/HEIF image support
python -m pip install pillow-avif-plugin  # AVIF image support
```

### TripoSR configuration (local mode)

- Local mode loads `stabilityai/TripoSR` through Hugging Face on first run.
- Ensure internet access for first model download and enough disk space for model cache.
- GPU is optional; CPU execution is supported but slower.
- If you use macOS Apple Silicon, install a compatible PyTorch build before running local reconstruction.

#### Local AI Setup

1. **Install PyTorch** — visit https://pytorch.org/get-started/locally/ and select the installation command for your OS and hardware. For CPU-only:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```
2. **Install TripoSR dependencies**:
   ```bash
   pip install transformers huggingface-hub
   ```
3. **First run** will automatically download the `stabilityai/TripoSR` model weights (~1.5 GB). Ensure you have a stable internet connection and sufficient disk space.
4. **GPU acceleration** (optional): Install CUDA-enabled PyTorch for faster inference. See the PyTorch install page for CUDA-specific commands.

### Meshy.ai configuration (cloud mode)

- Cloud mode requires a Meshy.ai API key.
- Obtain a key at https://www.meshy.ai and set it via environment variable or command argument.

```bash
export MESHY_API_KEY=your_key_here
```

### Desktop app build

```bash
dotnet build Image2STL.Desktop/Image2STL.Desktop.csproj
```

### Packaging targets

Use the included packaging helpers:

- Windows: `Image2STL.Desktop/packaging/build-windows-installer.ps1`
- macOS: `Image2STL.Desktop/packaging/build-macos-installer.sh`

## Desktop UI

The Avalonia desktop application provides a visual workflow for the full image-to-STL pipeline.

### Features

- **Image gallery** — Drag-and-drop or file picker to load 3-5 images. Thumbnails are displayed for each image; HEIC/HEIF files use a placeholder thumbnail.
- **3D preview** — Interactive wireframe viewer with mouse-drag rotation and scroll-wheel zoom.
- **Reconstruction mode** — Radio button toggle between Local (TripoSR) and Cloud (Meshy.ai) modes.
- **Scale controls** — Numeric input for target size in millimeters and dropdown for axis selection (longest/width/height/depth).
- **Generate 3D Model** — Starts the reconstruction pipeline with a progress bar and status messages.
- **Export STL** — Opens a save dialog (defaulting to `<ProjectName>.stl`) to export the final model.
- **Cancel** — Stops a running reconstruction operation.
- **Progress & warnings** — Real-time progress bar, processing time warnings (>10 min), and user-friendly error messages from the engine.
- **Drag-and-drop feedback** — Blue border highlight on the drop zone when files are dragged over the window.

### Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+N | New Project |
| Ctrl+O | Open Project |
| Ctrl+S | Save Project |
| Ctrl+I | Add Images |
| Ctrl+G | Generate 3D Model |

### Supported image formats

JPG, JPEG, PNG, HEIC, HEIF, WebP, AVIF

## Run tests

```bash
python -m unittest discover -s tests -v
```

## CLI examples

Create project:

```bash
python -m image2stl.cli new --base-dir /tmp --name MyHeadScan
```

Run reconstruction command:

```bash
python -m image2stl.cli run --json '{"command":"reconstruct","mode":"local","images":["a.jpg","b.png","c.heic"],"outputPath":"/tmp/model.obj"}'
```

Cloud mode (API key from environment):

```bash
MESHY_API_KEY=your_key_here python -m image2stl.cli run --json '{"command":"reconstruct","mode":"cloud","images":["a.jpg","b.png","c.heic"],"outputPath":"/tmp/model.obj"}'
```

Check image quality before reconstruction:

```bash
python -m image2stl.cli run --json '{"command":"check_images","images":["a.jpg","b.png","c.heic"]}'
```

Repair a mesh with target face count:

```bash
python -m image2stl.cli run --json '{"command":"repair","inputMesh":"raw.obj","outputMesh":"repaired.stl","targetFaceCount":100000}'
```
