#!/usr/bin/env bash
set -euo pipefail

PYTHON_CMD="${PYTHON_CMD:-python3}"
INSTALL_OPTIONAL_FORMATS=true
USE_SYSTEM_PYTHON=false
VENV_PATH="${VENV_PATH:-.venv}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<EOF
Usage: ./scripts/setup-macos.sh [--python <python_command>] [--venv-path <path>] [--system] [--skip-optional-formats]

Options:
  --python <python_command>   Python executable to use (default: python3 or PYTHON_CMD env var)
  --venv-path <path>          Virtual environment path (default: .venv)
  --system                    Install into system interpreter (adds --break-system-packages)
  --skip-optional-formats     Skip pillow-heif and pillow-avif-plugin
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_CMD="$2"
      shift 2
      ;;
    --venv-path)
      VENV_PATH="$2"
      shift 2
      ;;
    --system)
      USE_SYSTEM_PYTHON=true
      shift
      ;;
    --skip-optional-formats)
      INSTALL_OPTIONAL_FORMATS=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if ! command -v "$PYTHON_CMD" >/dev/null 2>&1; then
  echo "Error: Python command '$PYTHON_CMD' was not found on PATH."
  echo "Set a valid command with --python, e.g. --python /opt/homebrew/bin/python3"
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is required to install TripoSR from source but was not found."
  echo "Install Xcode Command Line Tools: xcode-select --install"
  exit 1
fi

cd "$PROJECT_ROOT"

if [[ "$USE_SYSTEM_PYTHON" == "true" ]]; then
  INSTALL_PYTHON="$PYTHON_CMD"
  PIP_FLAGS=(--break-system-packages)
  echo "Using system Python interpreter (not recommended):"
  "$INSTALL_PYTHON" -c "import sys; print(sys.executable); print(sys.version)"
else
  if [[ "$VENV_PATH" != /* ]]; then
    VENV_PATH="$PROJECT_ROOT/$VENV_PATH"
  fi
  if [[ ! -x "$VENV_PATH/bin/python" ]]; then
    echo "Creating virtual environment at: $VENV_PATH"
    "$PYTHON_CMD" -m venv "$VENV_PATH"
  fi
  INSTALL_PYTHON="$VENV_PATH/bin/python"
  PIP_FLAGS=()
  echo "Using virtual environment Python interpreter:"
  "$INSTALL_PYTHON" -c "import sys; print(sys.executable); print(sys.version)"
fi

if ! "$INSTALL_PYTHON" -m pip --version >/dev/null 2>&1; then
  echo "pip not found in target environment. Bootstrapping with ensurepip..."
  "$INSTALL_PYTHON" -m ensurepip --upgrade
fi

if ! "$INSTALL_PYTHON" -m pip --version >/dev/null 2>&1; then
  echo "Error: pip is still unavailable for $INSTALL_PYTHON"
  echo "This Python build may be incompatible with the current dependency set."
  echo "Install and use Python 3.11 (recommended) or 3.12, then rerun:"
  echo "  ./scripts/setup-macos.sh --python /opt/homebrew/bin/python3.11"
  exit 1
fi

pip_install() {
  if [[ ${#PIP_FLAGS[@]} -gt 0 ]]; then
    "$INSTALL_PYTHON" -m pip install "${PIP_FLAGS[@]}" "$@"
  else
    "$INSTALL_PYTHON" -m pip install "$@"
  fi
}

echo "Upgrading pip tooling..."
pip_install --upgrade pip setuptools wheel

echo "Installing core dependencies..."
pip_install pillow torch transformers huggingface-hub trimesh pymeshlab numpy

TRIPOSR_DIR="$PROJECT_ROOT/.vendor/TripoSR"
mkdir -p "$PROJECT_ROOT/.vendor"
if [[ ! -d "$TRIPOSR_DIR/.git" ]]; then
  echo "Cloning TripoSR source into $TRIPOSR_DIR..."
  git clone https://github.com/VAST-AI-Research/TripoSR.git "$TRIPOSR_DIR"
else
  echo "Updating existing TripoSR source checkout..."
  git -C "$TRIPOSR_DIR" pull --ff-only || true
fi

if [[ ! -f "$TRIPOSR_DIR/requirements.txt" ]]; then
  echo "Error: TripoSR requirements file not found at $TRIPOSR_DIR/requirements.txt"
  exit 1
fi

echo "Installing TripoSR runtime requirements..."
pip_install -r "$TRIPOSR_DIR/requirements.txt"

SITE_PACKAGES="$($INSTALL_PYTHON -c 'import site; print(site.getsitepackages()[0])')"
TRIPOSR_PTH="$SITE_PACKAGES/triposr_local.pth"
echo "Linking TripoSR source via .pth: $TRIPOSR_PTH"
printf '%s\n' "$TRIPOSR_DIR" > "$TRIPOSR_PTH"

if [[ "$INSTALL_OPTIONAL_FORMATS" == "true" ]]; then
  echo "Installing optional image format dependencies..."
  pip_install pillow-heif pillow-avif-plugin
fi

echo "Installing UI dependencies (PySide6 and other requirements)..."
pip_install -r "$PROJECT_ROOT/requirements.txt"

echo "Verifying required imports..."
"$INSTALL_PYTHON" - <<'PY'
import importlib
import sys

modules = [
    "torch",
    "PIL",
    "tsr",
    "transformers",
    "huggingface_hub",
    "trimesh",
    "pymeshlab",
    "numpy",
]
missing = []
for module in modules:
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append((module, str(exc)))

if missing:
    print("Missing or broken modules detected:")
    for module, err in missing:
        print(f" - {module}: {err}")
    sys.exit(1)

print("All required imports succeeded.")
PY

echo "Running Image2STL local environment check..."
"$INSTALL_PYTHON" -m image2stl.cli run --json '{"command":"check_environment","mode":"local"}'

echo
echo "Setup complete."
echo "Use this Python interpreter when running Image2STL: $INSTALL_PYTHON"
