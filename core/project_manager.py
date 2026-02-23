"""
Project lifecycle management (CRUD operations)
"""
import shutil
from pathlib import Path
from typing import List, Optional
import logging

from core.project import Project, PROJECT_FILE_NAME

logger = logging.getLogger(__name__)


class ProjectManager:
    """Manages project lifecycle"""

    def __init__(self, default_project_dir: Path = None):
        """Initialize project manager

        Args:
            default_project_dir: Default directory for projects
        """
        if default_project_dir is None:
            default_project_dir = Path.home() / "Documents" / "Image2STLProjects"

        self.default_project_dir = Path(default_project_dir)
        self.default_project_dir.mkdir(parents=True, exist_ok=True)

        self.current_project: Optional[Project] = None

    def create_project(self, name: str, project_dir: Path = None) -> Project:
        """Create a new project

        Args:
            name: Project name
            project_dir: Directory to create project in (uses default if None)

        Returns:
            New Project instance
        """
        if project_dir is None:
            project_dir = self.default_project_dir

        project = Project.create_new(name, project_dir)
        self.current_project = project
        return project

    def open_project(self, project_path: Path) -> Project:
        """Open an existing project

        Args:
            project_path: Path to project directory

        Returns:
            Loaded Project instance
        """
        project = Project.load(project_path)
        self.current_project = project
        return project

    def save_current_project(self):
        """Save the current project"""
        if self.current_project:
            self.current_project.save()
        else:
            logger.warning("No current project to save")

    def close_current_project(self):
        """Save and close the current project"""
        if self.current_project:
            self.save_current_project()
            logger.info(f"Closed project: {self.current_project.name}")
            self.current_project = None

    def list_projects(self, directory: Path = None) -> List[Path]:
        """List all projects in a directory

        Args:
            directory: Directory to search (uses default if None)

        Returns:
            List of project directory paths sorted by modification time (newest first)
        """
        if directory is None:
            directory = self.default_project_dir

        projects = []

        if directory.exists():
            for item in directory.iterdir():
                if item.is_dir() and (item / PROJECT_FILE_NAME).exists():
                    projects.append(item)

        return sorted(projects, key=lambda p: p.stat().st_mtime, reverse=True)

    def delete_project(self, project_path: Path):
        """Delete a project

        Args:
            project_path: Path to project directory
        """
        if project_path.exists():
            if self.current_project and self.current_project.project_path == project_path:
                self.current_project = None

            shutil.rmtree(project_path)
            logger.info(f"Deleted project: {project_path.name}")
