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
python -m pip install pillow torch transformers trimesh pymeshlab
```

### TripoSR configuration (local mode)

- Local mode loads `stabilityai/TripoSR` through Hugging Face on first run.
- Ensure internet access for first model download and enough disk space for model cache.
- GPU is optional; CPU execution is supported but slower.
- If you use macOS Apple Silicon, install a compatible PyTorch build before running local reconstruction.

### Desktop app build

```bash
dotnet build Image2STL.Desktop/Image2STL.Desktop.csproj
```

### Packaging targets

Use the included packaging helpers:

- Windows: `Image2STL.Desktop/packaging/build-windows-installer.ps1`
- macOS: `Image2STL.Desktop/packaging/build-macos-installer.sh`

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
