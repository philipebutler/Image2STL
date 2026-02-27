# Image2STL MVP

Image2STL converts 3-5 smartphone images into a printable STL workflow with:

- local reconstruction (TripoSR)
- cloud reconstruction (Meshy.ai)
- mesh repair/scaling commands
- PySide6 desktop UI with image gallery + 3D preview

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

### Python UI setup

```bash
python -m pip install -r requirements.txt
```

If local dependencies are reported missing, install them with the same interpreter you use to run the app:

```bash
<python-command> -m pip install pillow torch transformers huggingface-hub trimesh pymeshlab numpy
<python-command> -m pip install -r .vendor/TripoSR/requirements.txt
# plus source checkout linkage for tsr module (handled automatically by setup scripts)
```

### Launch UI

```bash
python main.py
```

### Features

- **Image gallery** — Drag-and-drop or file picker to load 3-5 images. Thumbnails are displayed for each image; HEIC/HEIF files show a placeholder. Images with available processed versions are highlighted with a green border and a "✓ processed" badge.
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
- **Settings** — Configure default scale, Meshy API key, and window preferences.
- **Foreground Isolation (preprocessing)** — Automatically or manually strip photo backgrounds before reconstruction using the `rembg` library. See [Foreground Isolation](#foreground-isolation) below.

### Keyboard shortcuts

| Shortcut | Action |
|----------|--------|
| Ctrl+N | New Project |
| Ctrl+O | Open Project |
| Ctrl+S | Save Project |
| Ctrl+I | Add Images |
| Ctrl+P | Isolate Foreground (manual preprocess) |
| Ctrl+Q | Quit |

### Supported image formats

JPG, JPEG, PNG, HEIC, HEIF, WebP, AVIF

### Foreground Isolation

Image2STL can automatically remove photo backgrounds before reconstruction.  This
typically improves 3D model quality by removing distracting background geometry.
The feature requires the optional `rembg` library:

```bash
pip install rembg
```

For the advanced **Fill holes** and **Island threshold** controls (under the
Advanced ▸ panel) to work, `scipy` is also required:

```bash
pip install scipy
```

If `scipy` is not installed those two features degrade gracefully — the controls
remain visible but have no effect.

#### UI controls (Foreground Isolation panel)

| Control | Description |
|---------|-------------|
| **Auto Isolate** checkbox | When checked, backgrounds are stripped automatically each time you click *Generate 3D Model* |
| **Run Preprocess** button | Manually run background removal on all loaded images right now |
| **Source** combo | Choose *Original* or *Processed* as the input for reconstruction. Switch to *Processed* after running preprocessing to use the cleaned images |
| **Strength** spin (0.0–1.0) | How aggressively the mask is cleaned up after background removal (0.5 is a good default) |
| **Advanced ▸** | Expand for fine-grained mask controls |
| — Fill holes | Fill small transparent holes that rembg leaves inside the foreground |
| — Island threshold | Minimum pixel area for a foreground region to survive (removes stray dots) |
| — Crop padding (px) | Extra pixels to leave around the tight foreground bounding box |

After preprocessing, each image tile in the gallery gains a green border and
"✓ processed" badge.  Processed images are cached under `preview/processed/`
inside your project directory using a deterministic name that encodes the source
file and current settings — re-running preprocessing with the same settings is
instant (cache hit).

#### Automatic vs manual mode

| | Automatic | Manual |
|-|-----------|--------|
| **How** | Check *Auto Isolate*, then click *Generate 3D Model* | Click *Run Preprocess* (or Images → Isolate Foreground / Ctrl+P) |
| **When** | Preprocessing runs before every reconstruction | Only when you explicitly trigger it |
| **Source** | Always uses the freshly-processed images for reconstruction | You control the *Source* selector — switch to *Processed* manually |
| **Best for** | Quick one-click workflow | Inspecting intermediate results or fine-tuning parameters |

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

### Foreground isolation CLI

Preprocess all images in a project (writes RGBA PNGs to `preview/processed/`):

```bash
python -m image2stl.cli preprocess-images --project-dir /path/to/MyProject
```

With custom settings:

```bash
python -m image2stl.cli preprocess-images \
  --project-dir /path/to/MyProject \
  --strength 0.7 \
  --island-threshold 200 \
  --crop-padding 20
```

Reconstruct using the processed image set:

```bash
python -m image2stl.cli reconstruct-project \
  --project-dir /path/to/MyProject \
  --preprocess-source processed
```

Auto-isolate and reconstruct in one command:

```bash
python -m image2stl.cli reconstruct-project \
  --project-dir /path/to/MyProject \
  --auto-isolate-foreground \
  --preprocess-strength 0.6
```

Reconstruct with assumption tuning (flat bottom + symmetry + preset policy):

```bash
python -m image2stl.cli reconstruct-project \
  --project-dir /path/to/MyProject \
  --assume-symmetry \
  --assumption-preset aggressive \
  --assumption-confidence 0.70
```

Assumption options:

- `--no-assumptions` disables assumption postprocessing.
- `--no-assume-flat-bottom` disables the flat-bottom correction.
- `--assume-symmetry` enables conservative symmetry correction.
- `--assumption-preset {conservative,standard,aggressive}` controls correction strength and safety limits.
- `--assumption-confidence <0..1>` sets the minimum confidence gate (effective threshold is max of this value and preset minimum).
