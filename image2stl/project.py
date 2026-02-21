from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .errors import SUPPORTED_IMAGE_EXTENSIONS


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class Project:
    projectId: str
    name: str
    created: str
    lastModified: str
    images: list[str] = field(default_factory=list)
    reconstructionMode: str = "local"
    modelPath: str = ""
    scaleMm: float = 150.0
    settings: dict = field(default_factory=lambda: {"meshQuality": "medium"})

    def save(self, project_dir: Path) -> Path:
        project_dir = project_dir.resolve()
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "images").mkdir(exist_ok=True)
        (project_dir / "models").mkdir(exist_ok=True)
        (project_dir / "preview").mkdir(exist_ok=True)
        self.lastModified = _now_iso()
        project_file = project_dir / "project.json"
        project_file.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return project_file

    def add_images(self, project_dir: Path, image_paths: list[Path]) -> list[str]:
        images_dir = (project_dir / "images").resolve()
        images_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for image_path in image_paths:
            src = image_path.resolve()
            if src.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image type: {src.suffix}")
            target = (images_dir / src.name).resolve()
            if images_dir not in target.parents:
                raise ValueError("Invalid image destination path")
            shutil.copy2(src, target)
            copied.append(f"images/{target.name}")
        self.images.extend(copied)
        self.lastModified = _now_iso()
        return copied


def create_project(base_dir: Path, name: str) -> tuple[Project, Path]:
    project_id = str(uuid4())
    project_dir = (base_dir / name).resolve()
    project = Project(
        projectId=project_id,
        name=name,
        created=_now_iso(),
        lastModified=_now_iso(),
    )
    project.save(project_dir)
    return project, project_dir


def load_project(project_dir: Path) -> Project:
    data = json.loads((project_dir / "project.json").read_text(encoding="utf-8"))
    return Project(**data)
