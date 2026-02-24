"""
Configuration management for Image2STL
"""
import copy
import json
from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


class Config:
    """Application configuration manager"""

    DEFAULT_CONFIG = {
        "app": {
            "name": "Image2STL",
            "version": "1.0.0",
            "last_project_dir": str(Path.home() / "Documents" / "Image2STLProjects"),
        },
        "defaults": {
            "scale_mm": 150.0,
            "reconstruction_mode": "local",
            "min_images": 3,
            "max_images": 5,
        },
        "triposr": {
            "model_path": "models/triposr",
            "device": "cpu",
            "chunk_size": 8192,
        },
        "meshy_api": {
            "base_url": "https://api.meshy.ai/v1",
            "timeout_seconds": 600,
            "api_key": "",
        },
        "mesh_repair": {
            "target_face_count": 100000,
            "fill_holes": True,
            "remove_internal_geometry": True,
            "min_wall_thickness_mm": 1.0,
        },
        "ui": {
            "theme": "default",
            "language": "en",
            "window_width": 1280,
            "window_height": 800,
            "show_grid": True,
            "show_axes": True,
        },
    }

    def __init__(self, config_path: str = None):
        """Initialize configuration

        Args:
            config_path: Path to config file. If None, uses default location.
        """
        if config_path is None:
            config_dir = Path.home() / ".image2stl"
            config_dir.mkdir(exist_ok=True)
            self.config_path = config_dir / "config.json"
        else:
            self.config_path = Path(config_path)

        self.config: Dict[str, Any] = {}
        self.load()

    def load(self):
        """Load configuration from file or create default"""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                self._merge_defaults()
            except Exception as e:
                logger.error(f"Failed to load config: {e}. Using defaults.")
                self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        else:
            logger.info("No config file found. Creating default.")
            self.config = copy.deepcopy(self.DEFAULT_CONFIG)
            self.save()

    def save(self):
        """Save configuration to file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation

        Args:
            key_path: Dot-separated path (e.g., "ui.window_width")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation

        Args:
            key_path: Dot-separated path (e.g., "ui.window_width")
            value: Value to set
        """
        keys = key_path.split(".")
        target = self.config

        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]

        target[keys[-1]] = value

    def _merge_defaults(self):
        """Merge loaded config with defaults to ensure all keys exist"""

        def merge_dict(default: dict, loaded: dict) -> dict:
            result = default.copy()
            for key, value in loaded.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result

        self.config = merge_dict(self.DEFAULT_CONFIG, self.config)
