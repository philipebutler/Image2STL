# Image2STL MVP

Image2STL converts 3-5 smartphone images into a printable STL workflow with:

- local reconstruction (TripoSR)
- cloud reconstruction (Meshy.ai)
- mesh repair/scaling commands
- Avalonia desktop image gallery + 3D preview shell

Implementation scope and behavior are defined in `SPEC.md`.

## Install

### One-command setup scripts (recommended)

macOS:

```bash
chmod +x scripts/setup-macos.sh
./scripts/setup-macos.sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-windows.ps1
```

Choose a specific interpreter when needed:

```bash
./scripts/setup-macos.sh --python /opt/homebrew/bin/python3
```

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup-windows.ps1 -PythonCommand "C:\Python311\python.exe"
```

Both scripts install required dependencies, clone TripoSR source to `.vendor/TripoSR`, wire it into the environment via a `.pth` file, verify imports, and run:

```bash
<python-command> -m image2stl.cli run --json '{"command":"check_environment","mode":"local"}'
```

By default, setup scripts create/use a project virtual environment at `.venv` and install there. This avoids macOS/Homebrew `externally-managed-environment` (PEP 668) errors.

Useful options:

- macOS: `./scripts/setup-macos.sh --python /opt/homebrew/bin/python3 --venv-path .venv`
- Windows: `powershell -ExecutionPolicy Bypass -File .\scripts\setup-windows.ps1 -PythonCommand py -VenvPath .venv`

Install into system Python only if you explicitly want that:

- macOS: `./scripts/setup-macos.sh --python /opt/homebrew/bin/python3 --system`
- Windows: `powershell -ExecutionPolicy Bypass -File .\scripts\setup-windows.ps1 -PythonCommand py -System`

### Python CLI dependencies

macOS:

```bash
python3 -m pip install pillow torch transformers trimesh pymeshlab numpy
```

Windows (PowerShell):

```powershell
py -m pip install pillow torch transformers trimesh pymeshlab numpy
```

If `py` is not available on Windows, use:

```bash
python -m pip install pillow torch transformers trimesh pymeshlab numpy
```

Optional format support:

macOS:

```bash
python3 -m pip install pillow-heif
python3 -m pip install pillow-avif-plugin
```

Windows (PowerShell):

```powershell
py -m pip install pillow-heif
py -m pip install pillow-avif-plugin
```

### TripoSR configuration (local mode)

- Local mode loads `stabilityai/TripoSR` through Hugging Face on first run.
- Ensure internet access for first model download and enough disk space for model cache.
- GPU is optional; CPU execution is supported but slower.
- If you use macOS Apple Silicon, install a compatible PyTorch build before running local reconstruction.

#### Local AI Setup

1. **Install PyTorch** — visit https://pytorch.org/get-started/locally/ and select the installation command for your OS and hardware.

   CPU-only examples:

   macOS:
   ```bash
   python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

   Windows (PowerShell):
   ```powershell
   py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

2. **Install TripoSR dependencies**:

   macOS:
   ```bash
   python3 -m pip install transformers huggingface-hub
   git clone https://github.com/VAST-AI-Research/TripoSR.git .vendor/TripoSR
   python3 -m pip install -r .vendor/TripoSR/requirements.txt
   SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
   echo "$PWD/.vendor/TripoSR" > "$SITE_PACKAGES/triposr_local.pth"
   ```

   Windows (PowerShell):
   ```powershell
   py -m pip install transformers huggingface-hub
   git clone https://github.com/VAST-AI-Research/TripoSR.git .vendor/TripoSR
   py -m pip install -r .\.vendor\TripoSR\requirements.txt
   $site = py -c "import site; print(site.getsitepackages()[0])"
   Set-Content -Path (Join-Path $site "triposr_local.pth") -Value (Resolve-Path .\.vendor\TripoSR)
   ```
3. **First run** will automatically download the `stabilityai/TripoSR` model weights (~1.5 GB). Ensure you have a stable internet connection and sufficient disk space.
    - During local reconstruction, engine progress now explicitly shows:
       - `Checking Python dependencies...`
       - `Checking TripoSR model cache...`
       - `Loading/downloading TripoSR model weights...`
4. **GPU acceleration** (optional): Install CUDA-enabled PyTorch for faster inference. See the PyTorch install page for CUDA-specific commands.

### Meshy.ai configuration (cloud mode)

- Cloud mode requires a Meshy.ai API key.
- Obtain a key at https://www.meshy.ai and set it via environment variable or command argument.

macOS/Linux:

```bash
export MESHY_API_KEY=your_key_here
```

Windows PowerShell (current session):

```powershell
$env:MESHY_API_KEY="your_key_here"
```

Windows PowerShell (persist for future sessions):

```powershell
setx MESHY_API_KEY "your_key_here"
```

### Desktop app build

```bash
dotnet build Image2STL.Desktop/Image2STL.Desktop.csproj
```

### Desktop setup checks (recommended)

The desktop app now includes built-in setup checks and Python interpreter selection:

- **Python Command** field — sets which Python executable the app uses (`python3`, `python`, or full path).
- **Check Local Setup** button — validates required local dependencies and TripoSR cache/download readiness.
- **Check Cloud Setup** button (Cloud mode) — validates Meshy.ai API key configuration.

Suggested values for **Python Command**:

- macOS: `.venv/bin/python` (recommended) or `/opt/homebrew/bin/python3`
- Windows: `.venv\Scripts\python.exe` (recommended) or full path like `C:\Python311\python.exe`

If local dependencies are reported missing, install them with the same interpreter configured in **Python Command**:

```bash
<python-command> -m pip install pillow torch transformers huggingface-hub trimesh pymeshlab numpy
<python-command> -m pip install -r .vendor/TripoSR/requirements.txt
# plus source checkout linkage for tsr module (handled automatically by setup scripts)
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
- **Cloud configuration** — In Cloud mode, enter Meshy.ai API key directly or specify the environment variable name to use.
- **Scale controls** — Numeric input for target size in millimeters and dropdown for axis selection (longest/width/height/depth).
- **Generate 3D Model** — Starts the reconstruction pipeline with a progress bar and status messages.
- **Setup checks** — `Check Local Setup` and `Check Cloud Setup` surface dependency/API-key issues in-app.
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

Check Python environment and TripoSR cache readiness (local mode):

```bash
python -m image2stl.cli run --json '{"command":"check_environment","mode":"local"}'
```

This returns:
- installed/missing required Python modules for local mode
- Python executable and version
- TripoSR cache status (`cached`, `not_cached`, or `unknown`) and whether a first-run download is likely required

Repair a mesh with target face count:

```bash
python -m image2stl.cli run --json '{"command":"repair","inputMesh":"raw.obj","outputMesh":"repaired.stl","targetFaceCount":100000}'
```
