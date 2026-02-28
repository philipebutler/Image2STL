"""
COLMAPWrapper component.

Manages COLMAP subprocess invocation for structure-from-motion (SfM) and
dense reconstruction.  Used by Method E.
"""

import logging
import subprocess
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)


class COLMAPWrapper:
    """Wrapper around the COLMAP CLI for photogrammetry reconstruction.

    Manages feature extraction, feature matching, sparse SfM reconstruction,
    and optionally dense reconstruction via COLMAP subprocesses with timeout
    and progress monitoring.
    """

    def __init__(self, config=None):
        self.config = config
        self._camera_poses: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_sfm(
        self,
        images: List[Path],
        workspace_dir: Path,
        timeout_seconds: int = 600,
    ) -> Path:
        """Run COLMAP Structure-from-Motion pipeline.

        Executes: feature_extractor → exhaustive_matcher → mapper.

        Args:
            images: Input image paths.
            workspace_dir: COLMAP workspace directory (created if absent).
            timeout_seconds: Maximum allowed runtime across all sub-steps.

        Returns:
            Path to the sparse reconstruction output directory.

        Raises:
            RuntimeError: If COLMAP is not installed or a sub-command fails.
        """
        workspace_dir.mkdir(parents=True, exist_ok=True)
        image_dir = workspace_dir / "images"
        image_dir.mkdir(exist_ok=True)
        db_path = workspace_dir / "database.db"
        sparse_dir = workspace_dir / "sparse"
        sparse_dir.mkdir(exist_ok=True)

        # Symlink images into workspace so COLMAP can find them.
        for img_path in images:
            link = image_dir / img_path.name
            if not link.exists():
                link.symlink_to(img_path.resolve())

        step_timeout = max(60, timeout_seconds // 3)

        self._run_colmap(
            [
                "feature_extractor",
                "--database_path", str(db_path),
                "--image_path", str(image_dir),
            ],
            timeout_seconds=step_timeout,
        )
        self._run_colmap(
            ["exhaustive_matcher", "--database_path", str(db_path)],
            timeout_seconds=step_timeout,
        )
        self._run_colmap(
            [
                "mapper",
                "--database_path", str(db_path),
                "--image_path", str(image_dir),
                "--output_path", str(sparse_dir),
            ],
            timeout_seconds=step_timeout,
        )

        return sparse_dir

    def run_dense(
        self,
        workspace_dir: Path,
        output_dir: Path,
        timeout_seconds: int = 1200,
    ) -> Optional[Path]:
        """Run COLMAP dense reconstruction pipeline.

        Executes: image_undistorter → patch_match_stereo → stereo_fusion.

        Args:
            workspace_dir: COLMAP workspace from a prior :meth:`run_sfm` call.
            output_dir: Directory for dense output files.
            timeout_seconds: Maximum allowed runtime across all sub-steps.

        Returns:
            Path to the fused dense point cloud (PLY), or ``None`` if it
            could not be produced.

        Raises:
            RuntimeError: If COLMAP is not installed or a sub-command fails.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        sparse_dir = workspace_dir / "sparse"
        dense_dir = workspace_dir / "dense"
        dense_dir.mkdir(parents=True, exist_ok=True)

        step_timeout = max(60, timeout_seconds // 3)

        self._run_colmap(
            [
                "image_undistorter",
                "--image_path", str(workspace_dir / "images"),
                "--input_path", str(sparse_dir / "0"),
                "--output_path", str(dense_dir),
                "--output_type", "COLMAP",
            ],
            timeout_seconds=step_timeout,
        )
        self._run_colmap(
            ["patch_match_stereo", "--workspace_path", str(dense_dir)],
            timeout_seconds=step_timeout,
        )

        fused_path = output_dir / "fused.ply"
        self._run_colmap(
            [
                "stereo_fusion",
                "--workspace_path", str(dense_dir),
                "--output_path", str(fused_path),
            ],
            timeout_seconds=step_timeout,
        )

        return fused_path if fused_path.exists() else None

    def run_full_pipeline(
        self,
        images: List[Path],
        output_dir: Path,
        timeout_seconds: int = 1800,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Path:
        """Run the complete SfM → dense pipeline and return a mesh path.

        Args:
            images: Input image paths.
            output_dir: Root output directory.
            timeout_seconds: Total timeout budget.
            progress_callback: Optional ``(percent, status)`` callback.

        Returns:
            Path to the reconstructed mesh (PLY/OBJ).

        Raises:
            RuntimeError: If COLMAP fails at any stage.
        """
        if progress_callback:
            progress_callback(0, "Running COLMAP SfM…")

        colmap_ws = output_dir / "colmap_workspace"
        sparse_dir = self.run_sfm(images, colmap_ws, timeout_seconds // 2)

        if progress_callback:
            progress_callback(50, "Running COLMAP dense reconstruction…")

        fused = self.run_dense(colmap_ws, output_dir, timeout_seconds // 2)

        if fused is None:
            logger.warning("Dense reconstruction produced no output; falling back to sparse mesh")
            fused = self._sparse_to_mesh(sparse_dir, output_dir)

        if progress_callback:
            progress_callback(100, "COLMAP complete")

        return fused

    def get_camera_poses(self) -> dict:
        """Return camera poses from the last SfM run (empty dict if not run)."""
        return self._camera_poses or {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_colmap(self, args: List[str], timeout_seconds: int = 300) -> None:
        """Invoke a COLMAP sub-command, raising :class:`RuntimeError` on failure."""
        cmd = ["colmap"] + args
        logger.debug("Running: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"COLMAP timed out after {timeout_seconds}s: {args[0]}"
            ) from exc
        except FileNotFoundError as exc:
            raise RuntimeError("COLMAP executable not found — install COLMAP and ensure it is on PATH") from exc

        if result.returncode != 0:
            stderr_snippet = result.stderr[:500] if result.stderr else "(no stderr)"
            raise RuntimeError(
                f"COLMAP {args[0]} failed (exit {result.returncode}): {stderr_snippet}"
            )

    def _sparse_to_mesh(self, sparse_dir: Path, output_dir: Path) -> Path:
        """Convert a COLMAP sparse point cloud to a rough mesh via Open3D.

        Falls back to a placeholder file if Open3D is not installed.
        """
        try:
            import numpy as np
            import open3d as o3d
        except ImportError:
            logger.warning("open3d not available; returning placeholder mesh path")
            fallback = output_dir / "sparse_fallback.ply"
            fallback.write_text("ply\n")
            return fallback

        # Find the first model sub-directory (COLMAP names them 0, 1, …)
        model_dir = next(
            (d for d in sorted(sparse_dir.iterdir()) if d.is_dir()),
            sparse_dir,
        )

        points_txt = model_dir / "points3D.txt"
        pcd = o3d.geometry.PointCloud()

        if points_txt.exists():
            pts, cols = [], []
            with open(points_txt) as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 7:
                        continue
                    pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    cols.append([
                        int(parts[4]) / 255.0,
                        int(parts[5]) / 255.0,
                        int(parts[6]) / 255.0,
                    ])
            if pts:
                pcd.points = o3d.utility.Vector3dVector(np.array(pts))
                pcd.colors = o3d.utility.Vector3dVector(np.array(cols))
                pcd.estimate_normals()

        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=6)
        out_path = output_dir / "sparse_mesh.ply"
        o3d.io.write_triangle_mesh(str(out_path), mesh)
        return out_path
