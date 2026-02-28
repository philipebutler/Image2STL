# Multi-Method 3D Reconstruction System with Intelligent Fallback

## Document Overview

This specification defines a production-ready 3D reconstruction system that attempts multiple methods in priority order (E → D → C) with intelligent fallback based on hardware capabilities, image quality, and reconstruction success.

**Reconstruction Methods:**

- **Method E (Primary)**: Hybrid Photogrammetry - Real images + AI synthesis + COLMAP
- **Method D (Secondary)**: Dust3R - True multi-view AI reconstruction
- **Method C (Tertiary)**: TripoSR Fusion - Multiple single-shot reconstructions aligned and fused

-----

## System Architecture

### High-Level Flow

```
User provides 3-5 images
    ↓
[Hardware Detection & Method Selection]
    ↓
┌─────────────────────────────────────────┐
│  Attempt Method E (Hybrid Photogrammetry)│
│  - Requires: GPU with 6GB+ VRAM         │
│  - Best quality                          │
└─────────────┬───────────────────────────┘
              │
         [Success?] ──Yes──→ [Return mesh]
              │
             No
              ↓
┌─────────────────────────────────────────┐
│  Attempt Method D (Dust3R)              │
│  - Requires: GPU with 4GB+ VRAM         │
│  - Good quality                          │
└─────────────┬───────────────────────────┘
              │
         [Success?] ──Yes──→ [Return mesh]
              │
             No
              ↓
┌─────────────────────────────────────────┐
│  Attempt Method C (TripoSR Fusion)      │
│  - Requires: CPU only                    │
│  - Acceptable quality                    │
└─────────────┬───────────────────────────┘
              │
         [Success?] ──Yes──→ [Return mesh]
              │
             No
              ↓
      [Report failure]
```

-----

## Detailed Architecture

### Project Structure Changes

```
head-scanner/
├── main.py
├── config.py
├── constants.py
│
├── core/
│   ├── __init__.py
│   ├── project.py
│   ├── project_manager.py
│   │
│   ├── reconstruction/                    # NEW: Reconstruction subsystem
│   │   ├── __init__.py
│   │   ├── base_reconstructor.py         # Abstract base class
│   │   ├── method_selector.py            # Hardware detection & method selection
│   │   ├── reconstruction_engine.py      # Orchestrator with fallback logic
│   │   │
│   │   ├── methods/                       # Individual reconstruction methods
│   │   │   ├── __init__.py
│   │   │   ├── method_e_hybrid.py        # Hybrid photogrammetry
│   │   │   ├── method_d_dust3r.py        # Dust3R multi-view
│   │   │   ├── method_c_triposr.py       # TripoSR fusion
│   │   │   └── method_cloud.py           # Meshy.ai (unchanged)
│   │   │
│   │   └── components/                    # Shared components
│   │       ├── __init__.py
│   │       ├── view_synthesizer.py       # SyncDreamer/Zero123
│   │       ├── colmap_wrapper.py         # COLMAP integration
│   │       ├── mesh_aligner.py           # ICP alignment
│   │       └── mesh_verifier.py          # Quality verification
│   │
│   ├── mesh_processor.py
│   └── image_processor.py
│
├── ui/
│   ├── main_window.py                     # UPDATED: Method selection UI
│   ├── widgets/
│   │   ├── control_panel.py              # UPDATED: Method display & override
│   │   ├── image_gallery.py
│   │   ├── viewer_3d.py
│   │   ├── progress_widget.py            # UPDATED: Multi-stage progress
│   │   └── method_status_widget.py       # NEW: Shows which method is running
│   └── dialogs/
│       ├── method_selection_dialog.py    # NEW: Manual method override
│       └── hardware_info_dialog.py       # NEW: Show detected hardware
│
└── requirements.txt                       # UPDATED: All method dependencies
```

-----

## Core Components

### 1. Hardware Detection & Method Selection

**File: `core/reconstruction/method_selector.py`**

```python
"""
Intelligent method selection based on available hardware
"""
import torch
import platform
import subprocess
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class ReconstructionMethod(Enum):
    """Available reconstruction methods"""
    METHOD_E = "hybrid_photogrammetry"  # E: Real + Synthetic + COLMAP
    METHOD_D = "dust3r"                 # D: Dust3R multi-view
    METHOD_C = "triposr_fusion"         # C: TripoSR fusion
    METHOD_CLOUD = "meshy_cloud"        # Cloud fallback

@dataclass
class HardwareCapabilities:
    """Detected hardware capabilities"""
    has_cuda: bool
    has_mps: bool  # Apple Metal Performance Shaders
    cuda_devices: List[str]
    total_vram_gb: float
    total_ram_gb: float
    cpu_cores: int
    platform: str
    
    @property
    def has_gpu(self) -> bool:
        """Check if any GPU acceleration available"""
        return self.has_cuda or self.has_mps
    
    @property
    def can_run_method_e(self) -> bool:
        """Method E requires 6GB+ VRAM"""
        return self.has_gpu and self.total_vram_gb >= 6.0
    
    @property
    def can_run_method_d(self) -> bool:
        """Method D requires 4GB+ VRAM"""
        return self.has_gpu and self.total_vram_gb >= 4.0
    
    @property
    def can_run_method_c(self) -> bool:
        """Method C runs on CPU (always available)"""
        return True

class MethodSelector:
    """
    Detects hardware and selects optimal reconstruction method
    """
    
    @staticmethod
    def detect_hardware() -> HardwareCapabilities:
        """
        Detect available hardware capabilities
        
        Returns:
            HardwareCapabilities object with all detected info
        """
        import psutil
        
        # Check CUDA
        has_cuda = torch.cuda.is_available()
        cuda_devices = []
        total_vram_gb = 0.0
        
        if has_cuda:
            num_devices = torch.cuda.device_count()
            for i in range(num_devices):
                props = torch.cuda.get_device_properties(i)
                device_name = props.name
                device_vram = props.total_memory / (1024**3)  # Convert to GB
                
                cuda_devices.append(f"{device_name} ({device_vram:.1f}GB)")
                total_vram_gb += device_vram
            
            logger.info(f"CUDA available: {num_devices} device(s), {total_vram_gb:.1f}GB total VRAM")
        
        # Check MPS (Apple Silicon)
        has_mps = False
        if platform.system() == "Darwin":  # macOS
            try:
                has_mps = torch.backends.mps.is_available()
                if has_mps:
                    # Estimate MPS memory (shared with system RAM)
                    # Apple Silicon typically dedicates ~50-70% of RAM to GPU
                    total_ram = psutil.virtual_memory().total / (1024**3)
                    total_vram_gb = total_ram * 0.6  # Conservative estimate
                    cuda_devices = [f"Apple Metal ({total_vram_gb:.1f}GB estimated)"]
                    logger.info(f"MPS available: {total_vram_gb:.1f}GB estimated")
            except:
                pass
        
        # System RAM
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # CPU cores
        cpu_cores = psutil.cpu_count(logical=False) or 1
        
        capabilities = HardwareCapabilities(
            has_cuda=has_cuda,
            has_mps=has_mps,
            cuda_devices=cuda_devices,
            total_vram_gb=total_vram_gb,
            total_ram_gb=total_ram_gb,
            cpu_cores=cpu_cores,
            platform=platform.system()
        )
        
        logger.info(f"Hardware detected: {capabilities}")
        
        return capabilities
    
    @staticmethod
    def select_method(
        capabilities: HardwareCapabilities,
        user_preference: Optional[ReconstructionMethod] = None,
        num_images: int = 3
    ) -> List[ReconstructionMethod]:
        """
        Select reconstruction method(s) with fallback chain
        
        Args:
            capabilities: Detected hardware capabilities
            user_preference: User's manual method selection (overrides auto)
            num_images: Number of input images
            
        Returns:
            Ordered list of methods to try (first = primary)
        """
        
        # If user specified a method, try that first
        if user_preference:
            methods = [user_preference]
            # Add standard fallback chain after user preference
            fallback_chain = MethodSelector._get_fallback_chain(capabilities)
            for method in fallback_chain:
                if method not in methods:
                    methods.append(method)
            return methods
        
        # Auto-select based on hardware
        methods = []
        
        # Try Method E if capable
        if capabilities.can_run_method_e:
            methods.append(ReconstructionMethod.METHOD_E)
            logger.info("Method E (Hybrid Photogrammetry) available - will try first")
        
        # Try Method D if capable
        if capabilities.can_run_method_d:
            methods.append(ReconstructionMethod.METHOD_D)
            logger.info("Method D (Dust3R) available - will try as fallback")
        
        # Method C always available (CPU fallback)
        methods.append(ReconstructionMethod.METHOD_C)
        logger.info("Method C (TripoSR Fusion) available - final fallback")
        
        # Cloud as ultimate fallback
        methods.append(ReconstructionMethod.METHOD_CLOUD)
        
        return methods
    
    @staticmethod
    def _get_fallback_chain(capabilities: HardwareCapabilities) -> List[ReconstructionMethod]:
        """Get standard fallback chain based on capabilities"""
        chain = []
        if capabilities.can_run_method_e:
            chain.append(ReconstructionMethod.METHOD_E)
        if capabilities.can_run_method_d:
            chain.append(ReconstructionMethod.METHOD_D)
        chain.append(ReconstructionMethod.METHOD_C)
        chain.append(ReconstructionMethod.METHOD_CLOUD)
        return chain
    
    @staticmethod
    def check_colmap_installed() -> bool:
        """
        Check if COLMAP is installed (required for Method E)
        
        Returns:
            True if COLMAP is available
        """
        try:
            result = subprocess.run(
                ["colmap", "--help"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    @staticmethod
    def get_method_requirements(method: ReconstructionMethod) -> dict:
        """
        Get requirements for a specific method
        
        Returns:
            Dictionary with requirements and recommendations
        """
        requirements = {
            ReconstructionMethod.METHOD_E: {
                "name": "Hybrid Photogrammetry",
                "min_vram_gb": 6.0,
                "recommended_vram_gb": 8.0,
                "requires_gpu": True,
                "requires_colmap": True,
                "estimated_time_seconds": 600,  # 10 minutes
                "quality": "Highest",
                "description": "Combines real photos with AI-generated views for photogrammetry"
            },
            ReconstructionMethod.METHOD_D: {
                "name": "Dust3R Multi-View",
                "min_vram_gb": 4.0,
                "recommended_vram_gb": 6.0,
                "requires_gpu": True,
                "requires_colmap": False,
                "estimated_time_seconds": 300,  # 5 minutes
                "quality": "High",
                "description": "AI model designed for sparse multi-view reconstruction"
            },
            ReconstructionMethod.METHOD_C: {
                "name": "TripoSR Fusion",
                "min_vram_gb": 0.0,
                "recommended_vram_gb": 0.0,
                "requires_gpu": False,
                "requires_colmap": False,
                "estimated_time_seconds": 180,  # 3 minutes
                "quality": "Good",
                "description": "Multiple single-shot reconstructions aligned and merged"
            },
            ReconstructionMethod.METHOD_CLOUD: {
                "name": "Meshy.ai Cloud",
                "min_vram_gb": 0.0,
                "recommended_vram_gb": 0.0,
                "requires_gpu": False,
                "requires_colmap": False,
                "estimated_time_seconds": 240,  # 4 minutes
                "quality": "High",
                "description": "Cloud-based reconstruction (requires API key and internet)"
            }
        }
        
        return requirements.get(method, {})
```

-----

### 2. Base Reconstructor Abstract Class

**File: `core/reconstruction/base_reconstructor.py`**

```python
"""
Abstract base class for all reconstruction methods
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ReconstructionResult:
    """Result from reconstruction attempt"""
    success: bool
    mesh_path: Optional[Path]
    method_used: str
    processing_time_seconds: float
    error_message: Optional[str] = None
    quality_score: Optional[float] = None  # 0-1, if available
    metadata: dict = None  # Method-specific metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseReconstructor(ABC):
    """
    Abstract base class for reconstruction methods
    
    All reconstruction methods must implement this interface
    """
    
    def __init__(self, config: 'Config'):
        """
        Initialize reconstructor
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.progress_callback: Optional[Callable[[int, str], None]] = None
    
    def set_progress_callback(self, callback: Callable[[int, str], None]):
        """
        Set callback for progress updates
        
        Args:
            callback: Function(progress: int, status: str)
        """
        self.progress_callback = callback
    
    def _update_progress(self, progress: int, status: str):
        """Update progress if callback set"""
        if self.progress_callback:
            self.progress_callback(progress, status)
        logger.info(f"{progress}%: {status}")
    
    @abstractmethod
    def can_run(self) -> tuple[bool, str]:
        """
        Check if this method can run on current system
        
        Returns:
            (can_run: bool, reason: str)
        """
        pass
    
    @abstractmethod
    def estimate_time(self, num_images: int) -> int:
        """
        Estimate processing time in seconds
        
        Args:
            num_images: Number of input images
            
        Returns:
            Estimated seconds
        """
        pass
    
    @abstractmethod
    def reconstruct(
        self,
        images: List[Path],
        output_dir: Path
    ) -> ReconstructionResult:
        """
        Perform reconstruction
        
        Args:
            images: List of input image paths
            output_dir: Directory for output files
            
        Returns:
            ReconstructionResult with success/failure info
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Get human-readable method name"""
        pass
    
    def validate_inputs(self, images: List[Path]) -> tuple[bool, str]:
        """
        Validate input images
        
        Args:
            images: List of image paths
            
        Returns:
            (is_valid: bool, error_message: str)
        """
        if not images:
            return False, "No images provided"
        
        if len(images) < 3:
            return False, "At least 3 images required"
        
        if len(images) > 5:
            return False, "Maximum 5 images supported"
        
        # Check all files exist
        for img_path in images:
            if not img_path.exists():
                return False, f"Image not found: {img_path}"
        
        return True, ""
```

-----

### 3. Method E: Hybrid Photogrammetry Implementation

**File: `core/reconstruction/methods/method_e_hybrid.py`**

```python
"""
Method E: Hybrid Photogrammetry
Real images + AI-synthesized views + COLMAP reconstruction
"""
import time
from pathlib import Path
from typing import List
import logging
import numpy as np
from PIL import Image

from core.reconstruction.base_reconstructor import BaseReconstructor, ReconstructionResult
from core.reconstruction.components.view_synthesizer import SyncDreamerSynthesizer
from core.reconstruction.components.colmap_wrapper import COLMAPWrapper
from core.reconstruction.components.mesh_verifier import MeshVerifier
from core.reconstruction.method_selector import MethodSelector

logger = logging.getLogger(__name__)

class MethodEHybrid(BaseReconstructor):
    """
    Method E: Hybrid Photogrammetry
    
    Workflow:
    1. Select best reference image from real photos
    2. Generate 16-20 synthetic views using SyncDreamer
    3. Combine real + synthetic images (20-25 total)
    4. Run COLMAP photogrammetry on combined set
    5. Verify geometry using real images
    6. Return high-quality mesh
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.view_synthesizer = None
        self.colmap = None
        self.verifier = MeshVerifier()
        
        # Configuration
        self.num_synthetic_views = config.get('method_e.num_synthetic_views', 16)
    
    def get_method_name(self) -> str:
        return "Hybrid Photogrammetry (E)"
    
    def can_run(self) -> tuple[bool, str]:
        """Check if Method E can run"""
        # Check COLMAP
        if not MethodSelector.check_colmap_installed():
            return False, "COLMAP not installed"
        
        # Check GPU
        import torch
        if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
            return False, "GPU required (no CUDA or MPS found)"
        
        # Check VRAM
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 6.0:
                return False, f"Insufficient VRAM ({vram_gb:.1f}GB, need 6GB+)"
        
        return True, "Ready"
    
    def estimate_time(self, num_images: int) -> int:
        """Estimate 10 minutes for Method E"""
        return 600
    
    def reconstruct(
        self,
        images: List[Path],
        output_dir: Path
    ) -> ReconstructionResult:
        """
        Run hybrid photogrammetry reconstruction
        """
        start_time = time.time()
        
        try:
            # Validate
            is_valid, error = self.validate_inputs(images)
            if not is_valid:
                return ReconstructionResult(
                    success=False,
                    mesh_path=None,
                    method_used=self.get_method_name(),
                    processing_time_seconds=time.time() - start_time,
                    error_message=error
                )
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # === STAGE 1: View Synthesis ===
            self._update_progress(10, "Initializing view synthesizer...")
            
            if self.view_synthesizer is None:
                self.view_synthesizer = SyncDreamerSynthesizer(self.config)
            
            self._update_progress(15, "Selecting reference image...")
            reference_img = self._select_best_reference(images)
            
            self._update_progress(20, f"Generating {self.num_synthetic_views} synthetic views...")
            synthetic_images = self.view_synthesizer.synthesize_multiple_views(
                reference_img,
                num_views=self.num_synthetic_views
            )
            
            # Save synthetic images
            synth_dir = output_dir / "synthetic_views"
            synth_dir.mkdir(exist_ok=True)
            
            synth_paths = []
            for i, synth_img in enumerate(synthetic_images):
                synth_path = synth_dir / f"synth_{i:03d}.jpg"
                Image.fromarray(synth_img).save(synth_path)
                synth_paths.append(synth_path)
                
                # Update progress during synthesis
                progress = 20 + int((i / len(synthetic_images)) * 15)
                self._update_progress(progress, f"Generated view {i+1}/{len(synthetic_images)}")
            
            # === STAGE 2: COLMAP Photogrammetry ===
            self._update_progress(35, "Initializing COLMAP...")
            
            if self.colmap is None:
                self.colmap = COLMAPWrapper(self.config)
            
            # Combine all images
            all_images = list(images) + synth_paths
            
            self._update_progress(40, f"Running photogrammetry on {len(all_images)} images...")
            
            raw_mesh_path = self.colmap.run_full_pipeline(
                all_images,
                output_dir,
                progress_callback=lambda p, s: self._update_progress(40 + int(p * 0.3), s)
            )
            
            # === STAGE 3: Verification ===
            self._update_progress(70, "Verifying geometry with real images...")
            
            # Get camera poses from COLMAP
            camera_poses = self.colmap.get_camera_poses()
            
            # Verify using only real images
            verified_mesh_path = self.verifier.verify_and_refine(
                raw_mesh_path,
                images,  # Only real images
                camera_poses,
                output_dir / "verified.obj"
            )
            
            # === STAGE 4: Quality Check ===
            self._update_progress(85, "Evaluating quality...")
            
            quality_score = self.verifier.compute_quality_score(
                verified_mesh_path,
                images,
                camera_poses
            )
            
            self._update_progress(100, "Complete!")
            
            processing_time = time.time() - start_time
            
            return ReconstructionResult(
                success=True,
                mesh_path=verified_mesh_path,
                method_used=self.get_method_name(),
                processing_time_seconds=processing_time,
                quality_score=quality_score,
                metadata={
                    'num_real_images': len(images),
                    'num_synthetic_images': len(synth_paths),
                    'colmap_registered_images': len(camera_poses)
                }
            )
            
        except Exception as e:
            logger.error(f"Method E failed: {e}", exc_info=True)
            
            return ReconstructionResult(
                success=False,
                mesh_path=None,
                method_used=self.get_method_name(),
                processing_time_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    def _select_best_reference(self, images: List[Path]) -> Path:
        """
        Select best image as reference for view synthesis
        
        Criteria:
        - Sharpness (Laplacian variance)
        - Contrast
        - Face detection confidence (if applicable)
        """
        import cv2
        
        best_score = -1
        best_image = images[0]
        
        for img_path in images:
            img = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Sharpness
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Contrast
            contrast = gray.std()
            
            # Combined score
            score = (laplacian_var / 1000.0) * 0.6 + (contrast / 100.0) * 0.4
            
            if score > best_score:
                best_score = score
                best_image = img_path
        
        logger.info(f"Selected {best_image.name} as reference (score: {best_score:.2f})")
        return best_image
```

-----

### 4. Method D: Dust3R Implementation

**File: `core/reconstruction/methods/method_d_dust3r.py`**

```python
"""
Method D: Dust3R Multi-View Reconstruction
True multi-view AI designed for sparse views (2-10 images)
"""
import time
from pathlib import Path
from typing import List
import logging
import torch
import numpy as np
from PIL import Image

from core.reconstruction.base_reconstructor import BaseReconstructor, ReconstructionResult

logger = logging.getLogger(__name__)

class MethodDDust3R(BaseReconstructor):
    """
    Method D: Dust3R
    
    Workflow:
    1. Load all 3-5 images
    2. Process pairwise geometric relationships
    3. Build global 3D scene representation
    4. Extract dense point cloud
    5. Convert to mesh via Poisson reconstruction
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.model = None
        self.device = None
    
    def get_method_name(self) -> str:
        return "Dust3R Multi-View (D)"
    
    def can_run(self) -> tuple[bool, str]:
        """Check if Method D can run"""
        # Check GPU
        if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
            return False, "GPU required (no CUDA or MPS found)"
        
        # Check VRAM
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 4.0:
                return False, f"Insufficient VRAM ({vram_gb:.1f}GB, need 4GB+)"
        
        return True, "Ready"
    
    def estimate_time(self, num_images: int) -> int:
        """Estimate 5 minutes for Method D"""
        return 300
    
    def reconstruct(
        self,
        images: List[Path],
        output_dir: Path
    ) -> ReconstructionResult:
        """
        Run Dust3R multi-view reconstruction
        """
        start_time = time.time()
        
        try:
            # Validate
            is_valid, error = self.validate_inputs(images)
            if not is_valid:
                return ReconstructionResult(
                    success=False,
                    mesh_path=None,
                    method_used=self.get_method_name(),
                    processing_time_seconds=time.time() - start_time,
                    error_message=error
                )
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # === STAGE 1: Initialize Model ===
            self._update_progress(10, "Loading Dust3R model...")
            
            if self.model is None:
                self._initialize_model()
            
            # === STAGE 2: Load Images ===
            self._update_progress(20, "Loading input images...")
            
            image_tensors = []
            for img_path in images:
                img = Image.open(img_path).convert('RGB')
                img_resized = img.resize((512, 512), Image.LANCZOS)
                img_array = np.array(img_resized).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                image_tensors.append(img_tensor)
            
            # Stack to batch
            batch = torch.stack(image_tensors).to(self.device)
            
            # === STAGE 3: Pairwise Reconstruction ===
            self._update_progress(30, "Computing pairwise geometry...")
            
            with torch.no_grad():
                # Process all pairs of images
                pairwise_results = []
                num_pairs = len(images) * (len(images) - 1) // 2
                pair_idx = 0
                
                for i in range(len(images)):
                    for j in range(i + 1, len(images)):
                        # Process pair (i, j)
                        pair = torch.stack([batch[i], batch[j]])
                        
                        result = self.model.forward(pair)
                        pairwise_results.append(result)
                        
                        pair_idx += 1
                        progress = 30 + int((pair_idx / num_pairs) * 20)
                        self._update_progress(
                            progress,
                            f"Processing pair {pair_idx}/{num_pairs}"
                        )
                
                # === STAGE 4: Global Scene ===
                self._update_progress(50, "Building global scene...")
                
                # Merge pairwise reconstructions into global scene
                global_scene = self.model.build_global_scene(pairwise_results)
                
                # === STAGE 5: Point Cloud Extraction ===
                self._update_progress(60, "Extracting point cloud...")
                
                # Get dense 3D points
                points_3d = global_scene['pts3d'].cpu().numpy()  # [N, 3]
                colors = global_scene.get('rgb', None)
                
                if colors is not None:
                    colors = colors.cpu().numpy()  # [N, 3]
                
                # === STAGE 6: Surface Reconstruction ===
                self._update_progress(70, "Reconstructing surface...")
                
                mesh_path = self._poisson_reconstruction(
                    points_3d,
                    colors,
                    output_dir / "dust3r_mesh.obj"
                )
            
            self._update_progress(100, "Complete!")
            
            processing_time = time.time() - start_time
            
            return ReconstructionResult(
                success=True,
                mesh_path=mesh_path,
                method_used=self.get_method_name(),
                processing_time_seconds=processing_time,
                metadata={
                    'num_points': len(points_3d),
                    'num_pairs_processed': len(pairwise_results)
                }
            )
            
        except Exception as e:
            logger.error(f"Method D failed: {e}", exc_info=True)
            
            return ReconstructionResult(
                success=False,
                mesh_path=None,
                method_used=self.get_method_name(),
                processing_time_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    def _initialize_model(self):
        """Initialize Dust3R model"""
        try:
            from dust3r.model import AsymmetricCroCo3DStereo
            
            # Set device
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                raise RuntimeError("No GPU available")
            
            # Load model
            model_path = self.config.get('method_d.model_path', 'naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt')
            
            self.model = AsymmetricCroCo3DStereo.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Dust3R model loaded on {self.device}")
            
        except ImportError:
            raise ImportError(
                "Dust3R not installed. Install with: pip install dust3r"
            )
    
    def _poisson_reconstruction(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        output_path: Path
    ) -> Path:
        """
        Convert point cloud to mesh using Poisson reconstruction
        
        Args:
            points: [N, 3] point coordinates
            colors: [N, 3] point colors (optional)
            output_path: Where to save mesh
            
        Returns:
            Path to mesh file
        """
        import open3d as o3d
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate normals
        pcd.estimate_normals()
        
        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(30)
        
        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=9,
            width=0,
            scale=1.1,
            linear_fit=False
        )
        
        # Remove low-density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Save mesh
        o3d.io.write_triangle_mesh(str(output_path), mesh)
        
        return output_path
```

-----

### 5. Method C: TripoSR Fusion Implementation

**File: `core/reconstruction/methods/method_c_triposr.py`**

```python
"""
Method C: TripoSR Fusion
Multiple single-shot reconstructions aligned and fused
"""
import time
from pathlib import Path
from typing import List
import logging
import torch
import numpy as np
from PIL import Image
import trimesh

from core.reconstruction.base_reconstructor import BaseReconstructor, ReconstructionResult
from core.reconstruction.components.mesh_aligner import MeshAligner

logger = logging.getLogger(__name__)

class MethodCTripoSR(BaseReconstructor):
    """
    Method C: TripoSR Fusion
    
    Workflow:
    1. Run TripoSR on each image independently (3-5 meshes)
    2. Align meshes using ICP (Iterative Closest Point)
    3. Fuse aligned meshes (average overlapping regions)
    4. Clean up and return final mesh
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.model = None
        self.device = None
        self.aligner = MeshAligner()
    
    def get_method_name(self) -> str:
        return "TripoSR Fusion (C)"
    
    def can_run(self) -> tuple[bool, str]:
        """Method C always works (CPU fallback)"""
        return True, "Ready (CPU)"
    
    def estimate_time(self, num_images: int) -> int:
        """Estimate ~40 seconds per image"""
        return 40 * num_images
    
    def reconstruct(
        self,
        images: List[Path],
        output_dir: Path
    ) -> ReconstructionResult:
        """
        Run TripoSR fusion reconstruction
        """
        start_time = time.time()
        
        try:
            # Validate
            is_valid, error = self.validate_inputs(images)
            if not is_valid:
                return ReconstructionResult(
                    success=False,
                    mesh_path=None,
                    method_used=self.get_method_name(),
                    processing_time_seconds=time.time() - start_time,
                    error_message=error
                )
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # === STAGE 1: Initialize Model ===
            self._update_progress(5, "Loading TripoSR model...")
            
            if self.model is None:
                self._initialize_model()
            
            # === STAGE 2: Reconstruct Each Image ===
            meshes = []
            individual_dir = output_dir / "individual_meshes"
            individual_dir.mkdir(exist_ok=True)
            
            for i, img_path in enumerate(images):
                progress_start = 10 + (i * 60 // len(images))
                progress_end = 10 + ((i + 1) * 60 // len(images))
                
                self._update_progress(
                    progress_start,
                    f"Reconstructing from image {i+1}/{len(images)}..."
                )
                
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img_tensor = self._preprocess_image(img)
                
                # Run TripoSR
                with torch.no_grad():
                    mesh = self.model.run(img_tensor)
                
                # Save individual mesh
                mesh_path = individual_dir / f"mesh_{i:02d}.obj"
                mesh.export(str(mesh_path))
                meshes.append(mesh)
                
                self._update_progress(
                    progress_end,
                    f"Completed mesh {i+1}/{len(images)}"
                )
            
            # === STAGE 3: Align Meshes ===
            self._update_progress(70, "Aligning meshes...")
            
            aligned_meshes = self.aligner.align_meshes(meshes)
            
            # === STAGE 4: Fuse Meshes ===
            self._update_progress(85, "Fusing meshes...")
            
            fused_mesh = self._fuse_meshes(aligned_meshes)
            
            # === STAGE 5: Cleanup ===
            self._update_progress(95, "Cleaning up mesh...")
            
            # Remove duplicate vertices, fill holes
            fused_mesh.merge_vertices()
            fused_mesh.fill_holes()
            fused_mesh.remove_degenerate_faces()
            
            # Save
            final_path = output_dir / "triposr_fused.obj"
            fused_mesh.export(str(final_path))
            
            self._update_progress(100, "Complete!")
            
            processing_time = time.time() - start_time
            
            return ReconstructionResult(
                success=True,
                mesh_path=final_path,
                method_used=self.get_method_name(),
                processing_time_seconds=processing_time,
                metadata={
                    'num_meshes_fused': len(meshes),
                    'final_vertices': len(fused_mesh.vertices),
                    'final_faces': len(fused_mesh.faces)
                }
            )
            
        except Exception as e:
            logger.error(f"Method C failed: {e}", exc_info=True)
            
            return ReconstructionResult(
                success=False,
                mesh_path=None,
                method_used=self.get_method_name(),
                processing_time_seconds=time.time() - start_time,
                error_message=str(e)
            )
    
    def _initialize_model(self):
        """Initialize TripoSR model"""
        try:
            from tsr.system import TSR
            
            # Set device (prefer GPU, fallback to CPU)
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
            
            # Load model
            model_path = self.config.get('method_c.model_path', 'stabilityai/TripoSR')
            
            self.model = TSR.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"TripoSR model loaded on {self.device}")
            
        except ImportError:
            raise ImportError(
                "TripoSR not installed. Install with: pip install tsr"
            )
    
    def _preprocess_image(self, img: Image.Image) -> torch.Tensor:
        """Preprocess image for TripoSR"""
        # Resize to 512x512
        img = img.resize((512, 512), Image.LANCZOS)
        
        # Convert to tensor
        img_array = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def _fuse_meshes(self, meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh:
        """
        Fuse multiple aligned meshes into one
        
        Strategy: Concatenate all vertices/faces, then merge duplicates
        """
        all_vertices = []
        all_faces = []
        vertex_offset = 0
        
        for mesh in meshes:
            all_vertices.append(mesh.vertices)
            all_faces.append(mesh.faces + vertex_offset)
            vertex_offset += len(mesh.vertices)
        
        # Concatenate
        combined_vertices = np.vstack(all_vertices)
        combined_faces = np.vstack(all_faces)
        
        # Create combined mesh
        fused = trimesh.Trimesh(
            vertices=combined_vertices,
            faces=combined_faces,
            process=False
        )
        
        return fused
```

-----

### 6. Reconstruction Engine with Fallback Logic

**File: `core/reconstruction/reconstruction_engine.py`**

```python
"""
Main reconstruction engine with intelligent fallback
"""
from pathlib import Path
from typing import List, Optional, Callable
import logging
from dataclasses import dataclass

from core.reconstruction.base_reconstructor import BaseReconstructor, ReconstructionResult
from core.reconstruction.method_selector import MethodSelector, ReconstructionMethod, HardwareCapabilities
from core.reconstruction.methods.method_e_hybrid import MethodEHybrid
from core.reconstruction.methods.method_d_dust3r import MethodDDust3R
from core.reconstruction.methods.method_c_triposr import MethodCTripoSR
from core.reconstruction.methods.method_cloud import MethodCloud  # Existing Meshy.ai
from core.mesh_processor import MeshProcessor
from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)

@dataclass
class MethodAttempt:
    """Record of a method attempt"""
    method: ReconstructionMethod
    result: ReconstructionResult

class ReconstructionEngine(QThread):
    """
    Orchestrates reconstruction with intelligent fallback
    
    Attempts methods in priority order (E → D → C → Cloud)
    Falls back automatically if a method fails
    """
    
    # Qt Signals
    progress_updated = Signal(int)  # 0-100
    status_updated = Signal(str)
    method_started = Signal(str)  # Method name
    method_completed = Signal(str, bool)  # Method name, success
    reconstruction_completed = Signal(str)  # Final mesh path
    reconstruction_failed = Signal(str, str)  # Error code, message
    
    def __init__(self, config: 'Config'):
        super().__init__()
        
        self.config = config
        self.mesh_processor = MeshProcessor(config)
        
        # Detect hardware once
        self.hardware = MethodSelector.detect_hardware()
        logger.info(f"Hardware capabilities: {self.hardware}")
        
        # Input parameters (set before starting thread)
        self.images: List[Path] = []
        self.output_dir: Path = None
        self.user_method_preference: Optional[ReconstructionMethod] = None
        
        # Attempt tracking
        self.attempts: List[MethodAttempt] = []
    
    def set_inputs(
        self,
        images: List[Path],
        output_dir: Path,
        user_preference: Optional[ReconstructionMethod] = None
    ):
        """
        Set reconstruction inputs
        
        Args:
            images: Input image paths
            output_dir: Output directory
            user_preference: User's manual method selection (optional)
        """
        self.images = images
        self.output_dir = output_dir
        self.user_method_preference = user_preference
    
    def run(self):
        """
        Main reconstruction workflow with fallback
        (Runs in background thread)
        """
        try:
            # Determine method priority
            methods_to_try = MethodSelector.select_method(
                self.hardware,
                self.user_method_preference,
                len(self.images)
            )
            
            logger.info(f"Will try methods in order: {[m.value for m in methods_to_try]}")
            
            # Try each method until one succeeds
            for method_enum in methods_to_try:
                reconstructor = self._get_reconstructor(method_enum)
                
                # Check if method can run
                can_run, reason = reconstructor.can_run()
                if not can_run:
                    logger.warning(f"Skipping {method_enum.value}: {reason}")
                    continue
                
                # Attempt reconstruction
                self.status_updated.emit(f"Attempting {reconstructor.get_method_name()}...")
                self.method_started.emit(reconstructor.get_method_name())
                
                result = self._attempt_method(reconstructor)
                
                # Record attempt
                self.attempts.append(MethodAttempt(method=method_enum, result=result))
                
                if result.success:
                    # Success! Process and return
                    self.method_completed.emit(reconstructor.get_method_name(), True)
                    
                    final_mesh = self._post_process(result.mesh_path)
                    
                    self.reconstruction_completed.emit(str(final_mesh))
                    return
                else:
                    # Failed, try next method
                    self.method_completed.emit(reconstructor.get_method_name(), False)
                    logger.warning(
                        f"{reconstructor.get_method_name()} failed: {result.error_message}"
                    )
                    
                    # Continue to next method in fallback chain
                    continue
            
            # All methods failed
            self._report_complete_failure()
            
        except Exception as e:
            logger.error(f"Reconstruction engine error: {e}", exc_info=True)
            self.reconstruction_failed.emit("UNKNOWN_ERROR", str(e))
    
    def _get_reconstructor(self, method: ReconstructionMethod) -> BaseReconstructor:
        """Get reconstructor instance for a method"""
        
        reconstructors = {
            ReconstructionMethod.METHOD_E: MethodEHybrid,
            ReconstructionMethod.METHOD_D: MethodDDust3R,
            ReconstructionMethod.METHOD_C: MethodCTripoSR,
            ReconstructionMethod.METHOD_CLOUD: MethodCloud
        }
        
        reconstructor_class = reconstructors[method]
        reconstructor = reconstructor_class(self.config)
        
        # Set progress callback
        reconstructor.set_progress_callback(self._on_method_progress)
        
        return reconstructor
    
    def _attempt_method(self, reconstructor: BaseReconstructor) -> ReconstructionResult:
        """Attempt reconstruction with a specific method"""
        
        logger.info(f"Attempting {reconstructor.get_method_name()}")
        
        result = reconstructor.reconstruct(self.images, self.output_dir)
        
        logger.info(
            f"{reconstructor.get_method_name()} {'succeeded' if result.success else 'failed'} "
            f"in {result.processing_time_seconds:.1f}s"
        )
        
        return result
    
    def _post_process(self, mesh_path: Path) -> Path:
        """
        Post-process successful mesh
        
        Steps:
        1. Mesh repair (make watertight)
        2. Optimization (simplify if needed)
        3. Scaling
        4. Export as STL
        """
        self.status_updated.emit("Post-processing mesh...")
        self.progress_updated.emit(90)
        
        # Repair
        repaired_path = self.output_dir / "repaired.obj"
        self.mesh_processor.repair_mesh(mesh_path, repaired_path)
        
        # Optimize
        optimized_path = self.output_dir / "optimized.obj"
        self.mesh_processor.optimize_mesh(repaired_path, optimized_path)
        
        # Scale (if configured)
        scale_mm = self.config.get('defaults.scale_mm', 150.0)
        final_path = self.output_dir / "final.stl"
        self.mesh_processor.scale_and_export(optimized_path, final_path, scale_mm)
        
        self.progress_updated.emit(100)
        
        return final_path
    
    def _on_method_progress(self, progress: int, status: str):
        """Callback for method progress updates"""
        self.progress_updated.emit(progress)
        self.status_updated.emit(status)
    
    def _report_complete_failure(self):
        """All methods failed - report to user"""
        
        error_messages = []
        for attempt in self.attempts:
            method_name = MethodSelector.get_method_requirements(attempt.method)['name']
            error_messages.append(
                f"{method_name}: {attempt.result.error_message}"
            )
        
        combined_message = "\n".join(error_messages)
        
        self.reconstruction_failed.emit(
            "ALL_METHODS_FAILED",
            f"All reconstruction methods failed:\n\n{combined_message}"
        )
```

-----

## UI Changes

### Updated Control Panel

**File: `ui/widgets/control_panel.py`** (Updated)

```python
"""
Control panel with method selection and status
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QComboBox, QGroupBox,
    QRadioButton, QButtonGroup
)
from PySide6.QtCore import Signal, Slot

from core.reconstruction.method_selector import MethodSelector, ReconstructionMethod

class ControlPanel(QWidget):
    """
    Control panel for reconstruction settings
    """
    
    # Signals
    generate_clicked = Signal()
    mode_changed = Signal(str)  # "auto" or specific method
    scale_changed = Signal(float)
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hardware = MethodSelector.detect_hardware()
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI"""
        layout = QVBoxLayout(self)
        
        # === Method Selection ===
        method_group = QGroupBox("Reconstruction Method")
        method_layout = QVBoxLayout(method_group)
        
        # Auto mode (default)
        self.auto_radio = QRadioButton("Automatic (Recommended)")
        self.auto_radio.setChecked(True)
        self.auto_radio.setToolTip(
            "Automatically select best method based on your hardware"
        )
        method_layout.addWidget(self.auto_radio)
        
        # Method E
        self.method_e_radio = QRadioButton("Hybrid Photogrammetry (Highest Quality)")
        can_run_e, reason_e = self._can_run_method(ReconstructionMethod.METHOD_E)
        self.method_e_radio.setEnabled(can_run_e)
        if not can_run_e:
            self.method_e_radio.setToolTip(f"Unavailable: {reason_e}")
        else:
            self.method_e_radio.setToolTip("Real + synthetic images + photogrammetry")
        method_layout.addWidget(self.method_e_radio)
        
        # Method D
        self.method_d_radio = QRadioButton("Dust3R Multi-View (High Quality)")
        can_run_d, reason_d = self._can_run_method(ReconstructionMethod.METHOD_D)
        self.method_d_radio.setEnabled(can_run_d)
        if not can_run_d:
            self.method_d_radio.setToolTip(f"Unavailable: {reason_d}")
        else:
            self.method_d_radio.setToolTip("AI designed for sparse views")
        method_layout.addWidget(self.method_d_radio)
        
        # Method C
        self.method_c_radio = QRadioButton("TripoSR Fusion (Good Quality, CPU)")
        self.method_c_radio.setToolTip("Multiple single-shot reconstructions fused")
        method_layout.addWidget(self.method_c_radio)
        
        # Cloud
        self.cloud_radio = QRadioButton("Cloud (Meshy.ai)")
        self.cloud_radio.setToolTip("Requires API key and internet connection")
        method_layout.addWidget(self.cloud_radio)
        
        # Button group
        self.method_buttons = QButtonGroup()
        self.method_buttons.addButton(self.auto_radio)
        self.method_buttons.addButton(self.method_e_radio)
        self.method_buttons.addButton(self.method_d_radio)
        self.method_buttons.addButton(self.method_c_radio)
        self.method_buttons.addButton(self.cloud_radio)
        
        layout.addWidget(method_group)
        
        # === Scale Settings ===
        scale_group = QGroupBox("Model Scale")
        scale_layout = QHBoxLayout(scale_group)
        
        scale_layout.addWidget(QLabel("Target size:"))
        
        self.scale_spinbox = QSpinBox()
        self.scale_spinbox.setRange(10, 500)
        self.scale_spinbox.setValue(self.config.get('defaults.scale_mm', 150))
        self.scale_spinbox.setSuffix(" mm")
        self.scale_spinbox.setToolTip("Size of longest dimension")
        scale_layout.addWidget(self.scale_spinbox)
        
        layout.addWidget(scale_group)
        
        # === Generate Button ===
        self.generate_button = QPushButton("Generate 3D Model")
        self.generate_button.setEnabled(False)
        self.generate_button.setMinimumHeight(50)
        self.generate_button.clicked.connect(self.generate_clicked)
        layout.addWidget(self.generate_button)
        
        # === Estimated Time ===
        self.time_label = QLabel()
        self.time_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.time_label)
        
        layout.addStretch()
        
        # Connect signals
        self.method_buttons.buttonClicked.connect(self._on_method_changed)
        self.scale_spinbox.valueChanged.connect(lambda v: self.scale_changed.emit(float(v)))
        
        # Update initial time estimate
        self._update_time_estimate()
    
    def _can_run_method(self, method: ReconstructionMethod) -> tuple[bool, str]:
        """Check if a method can run on this hardware"""
        # This would use actual method can_run() checks
        # For now, simplified version:
        
        if method == ReconstructionMethod.METHOD_E:
            if not self.hardware.can_run_method_e:
                if not self.hardware.has_gpu:
                    return False, "Requires GPU"
                else:
                    return False, f"Requires 6GB+ VRAM (have {self.hardware.total_vram_gb:.1f}GB)"
            if not MethodSelector.check_colmap_installed():
                return False, "COLMAP not installed"
            return True, ""
        
        elif method == ReconstructionMethod.METHOD_D:
            if not self.hardware.can_run_method_d:
                if not self.hardware.has_gpu:
                    return False, "Requires GPU"
                else:
                    return False, f"Requires 4GB+ VRAM (have {self.hardware.total_vram_gb:.1f}GB)"
            return True, ""
        
        elif method == ReconstructionMethod.METHOD_C:
            return True, ""  # Always available
        
        elif method == ReconstructionMethod.METHOD_CLOUD:
            api_key = self.config.get('meshy_api.api_key', '')
            if not api_key:
                return False, "API key not configured"
            return True, ""
        
        return False, "Unknown method"
    
    @Slot()
    def _on_method_changed(self):
        """Handle method selection change"""
        selected_method = self.get_selected_method()
        self.mode_changed.emit(selected_method)
        self._update_time_estimate()
    
    def get_selected_method(self) -> str:
        """Get selected method"""
        if self.auto_radio.isChecked():
            return "auto"
        elif self.method_e_radio.isChecked():
            return ReconstructionMethod.METHOD_E.value
        elif self.method_d_radio.isChecked():
            return ReconstructionMethod.METHOD_D.value
        elif self.method_c_radio.isChecked():
            return ReconstructionMethod.METHOD_C.value
        elif self.cloud_radio.isChecked():
            return ReconstructionMethod.METHOD_CLOUD.value
        return "auto"
    
    def _update_time_estimate(self, num_images: int = 3):
        """Update estimated processing time"""
        method = self.get_selected_method()
        
        if method == "auto":
            # Use fastest available method for estimate
            if self.hardware.can_run_method_d:
                time_seconds = 300
            elif self.hardware.can_run_method_e:
                time_seconds = 600
            else:
                time_seconds = 180
        else:
            # Get time for specific method
            reqs = MethodSelector.get_method_requirements(
                ReconstructionMethod(method)
            )
            time_seconds = reqs.get('estimated_time_seconds', 300)
        
        minutes = time_seconds // 60
        self.time_label.setText(f"Estimated time: ~{minutes} minutes")
    
    def update_generate_button(self, num_images: int):
        """Enable/disable generate button based on image count"""
        min_images = self.config.get('defaults.min_images', 3)
        max_images = self.config.get('defaults.max_images', 5)
        
        is_valid = min_images <= num_images <= max_images
        self.generate_button.setEnabled(is_valid)
        
        if num_images < min_images:
            self.generate_button.setToolTip(
                f"Add at least {min_images} images"
            )
        elif num_images > max_images:
            self.generate_button.setToolTip(
                f"Maximum {max_images} images"
            )
        else:
            self.generate_button.setToolTip("")
        
        self._update_time_estimate(num_images)
```

-----

## Summary

This specification provides:

1. **Complete fallback chain** (E → D → C → Cloud)
1. **Hardware detection** with VRAM checking
1. **Intelligent method selection** based on capabilities
1. **Full implementations** of all three primary methods
1. **UI integration** with method selection and status
1. **Error handling** with graceful fallback
1. **Progress tracking** through multi-stage reconstruction

**Total file count**: ~25 new/modified files
**Estimated implementation time**: 2-3 weeks
**Testing time**: 1 week

Would you like me to:

1. **Generate the remaining component files** (view synthesizer, COLMAP wrapper, mesh aligner)?
1. **Create installation/setup scripts** for all dependencies?
1. **Design the UI dialogs** (hardware info, method selection)?
1. **Write integration tests** for the fallback chain?