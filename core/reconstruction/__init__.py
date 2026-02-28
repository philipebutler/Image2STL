"""
Reconstruction subsystem for multi-method 3D reconstruction with intelligent fallback.

Provides:
- Hardware detection and method selection
- Abstract base class for reconstruction methods
- Standardized result types
"""

from core.reconstruction.method_selector import (
    ReconstructionMethod,
    HardwareCapabilities,
    MethodSelector,
)
from core.reconstruction.base_reconstructor import (
    BaseReconstructor,
    ReconstructionResult,
)

__all__ = [
    "ReconstructionMethod",
    "HardwareCapabilities",
    "MethodSelector",
    "BaseReconstructor",
    "ReconstructionResult",
]
