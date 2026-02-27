"""
Project data model
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)

PROJECT_FILE_NAME = "project.json"
PROJECT_SCHEMA_VERSION = "1.0"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".avif"}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class ProjectSettings:
    """Project-specific settings"""

    reconstruction_mode: str = "local"
    scale_mm: float = 150.0
    scale_axis: str = "longest"
    mesh_quality: str = "medium"
    # Foreground isolation / preprocessing
    auto_isolate_enabled: bool = False
    preprocess_strength: float = 0.5
    preprocess_source_mode: str = "original"
    hole_fill_enabled: bool = True
    island_removal_threshold: int = 500
    crop_padding: int = 10
    edge_feather_radius: int = 2
    contrast_strength: float = 0.0


@dataclass
class Project:
    """Project data model"""

    project_id: str
    name: str
    project_path: Path
    created: str
    last_modified: str
    images: List[str] = field(default_factory=list)
    processed_images: List[str] = field(default_factory=list)
    model_path: Optional[str] = None
    settings: ProjectSettings = field(default_factory=ProjectSettings)
    schema_version: str = PROJECT_SCHEMA_VERSION

    def __post_init__(self):
        if isinstance(self.project_path, str):
            self.project_path = Path(self.project_path)
        if isinstance(self.settings, dict):
            self.settings = ProjectSettings(**self.settings)

    @property
    def images_dir(self) -> Path:
        return self.project_path / "images"

    @property
    def models_dir(self) -> Path:
        return self.project_path / "models"

    @property
    def preview_dir(self) -> Path:
        return self.project_path / "preview"

    def get_image_paths(self) -> List[Path]:
        """Get absolute paths to all images"""
        return [self.project_path / img for img in self.images]

    def get_model_path(self) -> Optional[Path]:
        """Get absolute path to model"""
        if self.model_path:
            return self.project_path / self.model_path
        return None

    def add_image(self, source_path: Path) -> str:
        """Add image to project, copying it into the project images folder.

        Args:
            source_path: Path to source image file

        Returns:
            Relative path of added image
        """
        if source_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image type: {source_path.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
            )

        self.images_dir.mkdir(parents=True, exist_ok=True)

        dest_path = self.images_dir / source_path.name
        dest_path = dest_path.resolve()
        if dest_path.parent != self.images_dir.resolve():
            raise ValueError("Invalid image destination path")

        # Handle duplicate names
        if dest_path.exists():
            stem = source_path.stem
            suffix = source_path.suffix
            counter = 1
            max_attempts = 1000
            while dest_path.exists():
                if counter > max_attempts:
                    raise RuntimeError(
                        f"Unable to determine a unique filename for '{source_path.name}' after {max_attempts} attempts."
                    )
                dest_path = (self.images_dir / f"{stem}_{counter}{suffix}").resolve()
                if dest_path.parent != self.images_dir.resolve():
                    raise ValueError("Invalid image destination path")
                counter += 1

        shutil.copy2(source_path, dest_path)

        rel_path = str(dest_path.relative_to(self.project_path.resolve()))
        self.images.append(rel_path)
        self._update_modified()

        logger.info(f"Added image: {rel_path}")
        return rel_path

    def remove_image(self, image_path: str):
        """Remove image from project

        Args:
            image_path: Relative path to image
        """
        if image_path in self.images:
            self.images.remove(image_path)

            full_path = (self.project_path / image_path).resolve()
            project_root = self.project_path.resolve()
            # Only delete the file if it is inside the project directory
            try:
                full_path.relative_to(project_root)
                if full_path.exists():
                    full_path.unlink()
            except ValueError:
                logger.warning(f"Skipping deletion of path outside project: {full_path}")

            self._update_modified()
            logger.info(f"Removed image: {image_path}")

    def set_model(self, model_path: Path):
        """Set the final model path (absolute path converted to relative)"""
        resolved_model = model_path.resolve()
        project_root = self.project_path.resolve()
        models_root = self.models_dir.resolve()

        # Ensure the model path is within the models directory
        try:
            resolved_model.relative_to(models_root)
        except ValueError as exc:
            raise ValueError(f"Model path '{resolved_model}' must be inside the models directory '{models_root}'") from exc

        # Store the model path relative to the project root
        self.model_path = str(resolved_model.relative_to(project_root))
        self._update_modified()

    def _update_modified(self):
        self.last_modified = _now_iso()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "project_id": self.project_id,
            "name": self.name,
            "project_path": str(self.project_path),
            "created": self.created,
            "last_modified": self.last_modified,
            "images": self.images,
            "processed_images": self.processed_images,
            "model_path": self.model_path,
            "settings": asdict(self.settings),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        """Create Project from dictionary"""
        settings_data = data.get("settings", {})
        if isinstance(settings_data, dict):
            settings = ProjectSettings(**{
                k: v for k, v in settings_data.items()
                if k in ProjectSettings.__dataclass_fields__
            })
        else:
            settings = ProjectSettings()

        return cls(
            project_id=data["project_id"],
            name=data["name"],
            project_path=Path(data["project_path"]),
            created=data["created"],
            last_modified=data["last_modified"],
            images=data.get("images", []),
            processed_images=data.get("processed_images", []),
            model_path=data.get("model_path"),
            settings=settings,
            schema_version=data.get("schema_version", PROJECT_SCHEMA_VERSION),
        )

    def save(self):
        """Save project to disk"""
        project_file = self.project_path / PROJECT_FILE_NAME
        try:
            with open(project_file, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Saved project: {self.name}")
        except Exception as e:
            logger.error(f"Failed to save project: {e}")
            raise

    @classmethod
    def load(cls, project_path: Path) -> "Project":
        """Load project from disk

        Args:
            project_path: Path to project directory

        Returns:
            Project instance
        """
        project_file = project_path / PROJECT_FILE_NAME

        if not project_file.exists():
            raise FileNotFoundError(f"Project file not found: {project_file}")

        try:
            with open(project_file, "r") as f:
                data = json.load(f)
            project = cls.from_dict(data)
            logger.info(f"Loaded project: {project.name}")
            return project
        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            raise

    @classmethod
    def create_new(cls, name: str, project_dir: Path) -> "Project":
        """Create a new project

        Args:
            name: Project name
            project_dir: Parent directory for project

        Returns:
            New Project instance
        """
        project_path = (project_dir / name).resolve()
        project_path.mkdir(parents=True, exist_ok=True)
        (project_path / "images").mkdir(exist_ok=True)
        (project_path / "models").mkdir(exist_ok=True)
        (project_path / "preview").mkdir(exist_ok=True)

        now = _now_iso()
        project = cls(
            project_id=str(uuid4()),
            name=name,
            project_path=project_path,
            created=now,
            last_modified=now,
        )
        project.save()

        logger.info(f"Created new project: {name}")
        return project
