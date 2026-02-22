# Image2STL MVP

This repository now contains a minimal MVP implementation driven by `SPEC.md`.

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Build desktop UI

```bash
dotnet build Image2STL.Desktop/Image2STL.Desktop.csproj
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
