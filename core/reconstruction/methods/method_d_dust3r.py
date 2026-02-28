"""
Method D: Dust3R Multi-View reconstruction.

Uses the Dust3R AI model for pairwise reconstruction followed by global
scene assembly and Poisson surface reconstruction.  Requires 4+ GB VRAM.
"""

import time
from pathlib import Path
from typing import List, Optional

from core.reconstruction.base_reconstructor import BaseReconstructor, ReconstructionResult
from core.reconstruction.method_selector import MethodSelector, ReconstructionMethod


class MethodDDust3R(BaseReconstructor):
    """Dust3R multi-view reconstruction method.

    Performs pairwise depth estimation across all image pairs, builds a
    global point cloud, then reconstructs the surface with Poisson meshing.

    The Dust3R model is loaded lazily on the first :meth:`reconstruct` call
    so that startup time remains fast.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._device: Optional[str] = None

    def can_run(self) -> tuple:
        """Check whether Method D can run on the current hardware.

        Returns:
            ``(can_run: bool, reason: str)``
        """
        hw = MethodSelector.detect_hardware()
        if not hw.can_run_method_d:
            if not hw.has_gpu:
                return False, "Method D requires a GPU with 4+ GB VRAM (no GPU detected)"
            return (
                False,
                f"Method D requires 4+ GB VRAM (detected {hw.total_vram_gb:.1f} GB)",
            )
        return True, "Ready"

    def estimate_time(self, num_images: int) -> int:
        """Estimate processing time for Method D.

        Args:
            num_images: Number of input images.

        Returns:
            Estimated seconds (scales with image pairs O(n²)).
        """
        base = MethodSelector.get_method_requirements(ReconstructionMethod.METHOD_D)[
            "estimated_time_seconds"
        ]
        pairs = max(1, num_images * (num_images - 1) // 2)
        return base + pairs * 15

    def reconstruct(self, images: List[Path], output_dir: Path) -> ReconstructionResult:
        """Run Dust3R multi-view reconstruction.

        Pipeline:
        1. Validate inputs.
        2. Load Dust3R model (lazy).
        3. Load and preprocess images.
        4. Compute pairwise geometry.
        5. Build global scene.
        6. Extract point cloud and run Poisson surface reconstruction.

        Args:
            images: Input image paths.
            output_dir: Output directory.

        Returns:
            :class:`ReconstructionResult` with success/failure details.
        """
        start = time.time()

        is_valid, error = self.validate_inputs(images)
        if not is_valid:
            return ReconstructionResult(
                success=False,
                mesh_path=None,
                method_used=self.get_method_name(),
                processing_time_seconds=time.time() - start,
                error_message=error,
            )

        try:
            import numpy as np
            import torch
            from PIL import Image as PILImage

            output_dir.mkdir(parents=True, exist_ok=True)

            # Stage 1: Load model
            self._update_progress(10, "Loading Dust3R model…")
            if self._model is None:
                self._initialize_model()

            # Stage 2: Load images
            self._update_progress(20, "Loading input images…")
            image_tensors = []
            for img_path in images:
                img = PILImage.open(str(img_path)).convert("RGB").resize((512, 512))
                arr = np.array(img).astype("float32") / 255.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1)
                image_tensors.append(tensor)

            batch = torch.stack(image_tensors).to(self._device)

            # Stage 3: Pairwise reconstruction
            self._update_progress(30, "Computing pairwise geometry…")
            pairwise_results = []
            num_pairs = len(images) * (len(images) - 1) // 2
            pair_idx = 0

            with torch.no_grad():
                for i in range(len(images)):
                    for j in range(i + 1, len(images)):
                        pair = torch.stack([batch[i], batch[j]])
                        result = self._model.forward(pair)
                        pairwise_results.append(result)
                        pair_idx += 1
                        progress = 30 + int((pair_idx / max(1, num_pairs)) * 20)
                        self._update_progress(
                            progress,
                            f"Processing pair {pair_idx}/{num_pairs}",
                        )

                # Stage 4: Global scene
                self._update_progress(50, "Building global scene…")
                global_scene = self._model.build_global_scene(pairwise_results)

                # Stage 5: Point cloud
                self._update_progress(60, "Extracting point cloud…")
                points_3d = global_scene["pts3d"].cpu().numpy()
                colors = global_scene.get("rgb", None)
                if colors is not None:
                    colors = colors.cpu().numpy()

            # Stage 6: Poisson surface reconstruction
            self._update_progress(70, "Reconstructing surface…")
            mesh_path = self._poisson_reconstruction(
                points_3d,
                colors,
                output_dir / "dust3r_mesh.obj",
            )
            self._update_progress(100, "Complete!")

            return ReconstructionResult(
                success=True,
                mesh_path=mesh_path,
                method_used=self.get_method_name(),
                processing_time_seconds=time.time() - start,
                metadata={
                    "num_points": len(points_3d),
                    "num_pairs_processed": len(pairwise_results),
                },
            )

        except Exception as exc:
            return ReconstructionResult(
                success=False,
                mesh_path=None,
                method_used=self.get_method_name(),
                processing_time_seconds=time.time() - start,
                error_message=str(exc),
            )

    def get_method_name(self) -> str:
        return "Dust3R Multi-View (Method D)"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initialize_model(self) -> None:
        """Lazy-load the Dust3R model.

        Raises:
            ImportError: If the ``dust3r`` package is not installed.
        """
        try:
            import torch
            from dust3r.model import AsymmetricCroCo3DStereo  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Dust3R is not installed. "
                "Install with: pip install dust3r  (or from https://github.com/naver/dust3r)"
            ) from exc

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            raise RuntimeError("No GPU available — Method D requires a GPU")

        model_path = (
            self.config.get(
                "method_d.model_path",
                "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt",
            )
            if self.config else "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        )
        self._model = AsymmetricCroCo3DStereo.from_pretrained(model_path)
        self._model.to(self._device)
        self._model.eval()

    def _poisson_reconstruction(
        self,
        points,
        colors,
        output_path: Path,
    ) -> Path:
        """Convert a point cloud to a mesh using Poisson surface reconstruction.

        Args:
            points: ``(N, 3)`` point coordinates array.
            colors: ``(N, 3)`` point colours, or ``None``.
            output_path: Destination mesh path.

        Returns:
            *output_path* after writing the mesh.

        Raises:
            ImportError: If ``open3d`` is not installed.
        """
        try:
            import numpy as np
            import open3d as o3d
        except ImportError as exc:
            raise ImportError(
                "open3d is required for Poisson reconstruction — install with: pip install open3d"
            ) from exc

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(30)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=9,
            width=0,
            scale=1.1,
            linear_fit=False,
        )

        import numpy as np
        vertices_to_remove = np.asarray(densities) < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        o3d.io.write_triangle_mesh(str(output_path), mesh)
        return output_path
