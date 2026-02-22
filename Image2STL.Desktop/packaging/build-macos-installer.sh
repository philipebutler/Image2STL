#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

dotnet publish "$ROOT_DIR/Image2STL.Desktop.csproj" -c Release -r osx-arm64 --self-contained true /p:PublishSingleFile=true

PUBLISH_DIR="$ROOT_DIR/bin/Release/net10.0/osx-arm64/publish"
APP_NAME="Image2STL.app"

if command -v create-dmg >/dev/null 2>&1 && [ -d "$PUBLISH_DIR/$APP_NAME" ]; then
  create-dmg --overwrite "Image2STL.dmg" "$PUBLISH_DIR/$APP_NAME"
else
  echo "Publish completed at $PUBLISH_DIR (create-dmg not installed or app bundle missing)."
fi
