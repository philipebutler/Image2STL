"""
Intelligent method selection based on available hardware.

Detects GPU capabilities (CUDA/MPS), VRAM, and system resources to determine
which reconstruction methods can run, and builds a prioritised fallback chain.
"""

import platform
import subprocess
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class ReconstructionMethod(Enum):
    """Available reconstruction methods."""

    METHOD_E = "hybrid_photogrammetry"  # Real + Synthetic + COLMAP
    METHOD_D = "dust3r"                 # Dust3R multi-view
    METHOD_C = "triposr_fusion"         # TripoSR fusion
    METHOD_CLOUD = "meshy_cloud"        # Meshy.ai cloud


@dataclass
class HardwareCapabilities:
    """Detected hardware capabilities."""

    has_cuda: bool
    has_mps: bool
    cuda_devices: List[str] = field(default_factory=list)
    total_vram_gb: float = 0.0
    total_ram_gb: float = 0.0
    cpu_cores: int = 1
    platform: str = ""

    @property
    def has_gpu(self) -> bool:
        """Check if any GPU acceleration is available."""
        return self.has_cuda or self.has_mps

    @property
    def can_run_method_e(self) -> bool:
        """Method E requires 6 GB+ VRAM."""
        return self.has_gpu and self.total_vram_gb >= 6.0

    @property
    def can_run_method_d(self) -> bool:
        """Method D requires 4 GB+ VRAM."""
        return self.has_gpu and self.total_vram_gb >= 4.0

    @property
    def can_run_method_c(self) -> bool:
        """Method C runs on CPU (always available)."""
        return True


class MethodSelector:
    """Detects hardware and selects optimal reconstruction method."""

    @staticmethod
    def detect_hardware() -> HardwareCapabilities:
        """Detect available hardware capabilities.

        Attempts to import ``torch`` and ``psutil`` for detailed detection.
        Falls back gracefully when either library is unavailable.
        """
        has_cuda = False
        has_mps = False
        cuda_devices: List[str] = []
        total_vram_gb = 0.0
        total_ram_gb = 0.0
        cpu_cores = 1

        # --- torch-based GPU detection ---
        try:
            import torch  # type: ignore

            has_cuda = torch.cuda.is_available()
            if has_cuda:
                num_devices = torch.cuda.device_count()
                for i in range(num_devices):
                    props = torch.cuda.get_device_properties(i)
                    device_vram = props.total_memory / (1024 ** 3)
                    cuda_devices.append(f"{props.name} ({device_vram:.1f}GB)")
                    total_vram_gb += device_vram
                logger.info(
                    "CUDA available: %d device(s), %.1fGB total VRAM",
                    num_devices,
                    total_vram_gb,
                )

            if platform.system() == "Darwin":
                try:
                    has_mps = torch.backends.mps.is_available()
                except Exception:
                    pass
        except ImportError:
            logger.debug("torch not available — GPU detection skipped")

        # --- MPS VRAM estimation ---
        if has_mps and total_vram_gb == 0.0:
            try:
                import psutil  # type: ignore

                total_ram = psutil.virtual_memory().total / (1024 ** 3)
                total_vram_gb = total_ram * 0.6  # conservative estimate
                cuda_devices = [f"Apple Metal ({total_vram_gb:.1f}GB estimated)"]
                logger.info("MPS available: %.1fGB estimated", total_vram_gb)
            except ImportError:
                pass

        # --- System RAM / CPU ---
        try:
            import psutil  # type: ignore

            total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            cpu_cores = psutil.cpu_count(logical=False) or 1
        except ImportError:
            import os

            cpu_cores = os.cpu_count() or 1

        capabilities = HardwareCapabilities(
            has_cuda=has_cuda,
            has_mps=has_mps,
            cuda_devices=cuda_devices,
            total_vram_gb=total_vram_gb,
            total_ram_gb=total_ram_gb,
            cpu_cores=cpu_cores,
            platform=platform.system(),
        )

        logger.info("Hardware detected: %s", capabilities)
        return capabilities

    @staticmethod
    def select_method(
        capabilities: HardwareCapabilities,
        user_preference: Optional[ReconstructionMethod] = None,
        num_images: int = 3,
    ) -> List[ReconstructionMethod]:
        """Select reconstruction method(s) with fallback chain.

        Args:
            capabilities: Detected hardware capabilities.
            user_preference: User's manual method selection (overrides auto).
            num_images: Number of input images.

        Returns:
            Ordered list of methods to try (first = primary).
        """
        if user_preference:
            methods = [user_preference]
            for method in MethodSelector._get_fallback_chain(capabilities):
                if method not in methods:
                    methods.append(method)
            return methods

        return MethodSelector._get_fallback_chain(capabilities)

    @staticmethod
    def _get_fallback_chain(
        capabilities: HardwareCapabilities,
    ) -> List[ReconstructionMethod]:
        """Build standard fallback chain based on capabilities."""
        chain: List[ReconstructionMethod] = []
        if capabilities.can_run_method_e:
            chain.append(ReconstructionMethod.METHOD_E)
        if capabilities.can_run_method_d:
            chain.append(ReconstructionMethod.METHOD_D)
        chain.append(ReconstructionMethod.METHOD_C)
        chain.append(ReconstructionMethod.METHOD_CLOUD)
        return chain

    @staticmethod
    def check_colmap_installed() -> bool:
        """Check if COLMAP is installed (required for Method E)."""
        try:
            result = subprocess.run(
                ["colmap", "--help"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    @staticmethod
    def get_method_requirements(method: ReconstructionMethod) -> dict:
        """Get requirements and metadata for a specific method."""
        requirements = {
            ReconstructionMethod.METHOD_E: {
                "name": "Hybrid Photogrammetry",
                "min_vram_gb": 6.0,
                "recommended_vram_gb": 8.0,
                "requires_gpu": True,
                "requires_colmap": True,
                "estimated_time_seconds": 600,
                "quality": "Highest",
                "description": "Combines real photos with AI-generated views for photogrammetry",
            },
            ReconstructionMethod.METHOD_D: {
                "name": "Dust3R Multi-View",
                "min_vram_gb": 4.0,
                "recommended_vram_gb": 6.0,
                "requires_gpu": True,
                "requires_colmap": False,
                "estimated_time_seconds": 300,
                "quality": "High",
                "description": "AI model designed for sparse multi-view reconstruction",
            },
            ReconstructionMethod.METHOD_C: {
                "name": "TripoSR Fusion",
                "min_vram_gb": 0.0,
                "recommended_vram_gb": 0.0,
                "requires_gpu": False,
                "requires_colmap": False,
                "estimated_time_seconds": 180,
                "quality": "Good",
                "description": "Multiple single-shot reconstructions aligned and merged",
            },
            ReconstructionMethod.METHOD_CLOUD: {
                "name": "Meshy.ai Cloud",
                "min_vram_gb": 0.0,
                "recommended_vram_gb": 0.0,
                "requires_gpu": False,
                "requires_colmap": False,
                "estimated_time_seconds": 240,
                "quality": "High",
                "description": "Cloud-based reconstruction (requires API key and internet)",
            },
        }
        return requirements.get(method, {})
