"""
Reconstruction subsystem for multi-method 3D reconstruction with intelligent fallback.

Provides:
- Hardware detection and method selection
- Abstract base class for reconstruction methods
- Standardized result types
- Concrete method stubs (E, D, C, Cloud)
- Shared component stubs (ViewSynthesizer, COLMAPWrapper, MeshAligner, MeshVerifier)
- Orchestration engine with fallback and post-processing (Phase 3)
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
from core.reconstruction.methods import (
    MethodEHybrid,
    MethodDDust3R,
    MethodCTripoSR,
    MethodCloud,
)
from core.reconstruction.components import (
    ViewSynthesizer,
    COLMAPWrapper,
    MeshAligner,
    MeshVerifier,
)
from core.reconstruction.engine import ReconstructionEngine, MethodAttempt

__all__ = [
    "ReconstructionMethod",
    "HardwareCapabilities",
    "MethodSelector",
    "BaseReconstructor",
    "ReconstructionResult",
    "MethodEHybrid",
    "MethodDDust3R",
    "MethodCTripoSR",
    "MethodCloud",
    "ViewSynthesizer",
    "COLMAPWrapper",
    "MeshAligner",
    "MeshVerifier",
    "ReconstructionEngine",
    "MethodAttempt",
]
