"""
Image processing pipeline for forensic analysis.
Orchestrates multiple processing steps and manages output files.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List

import cv2
import numpy as np

from app.core.hashing import calculate_data_sha256
from app.core.metadata import ProcessingMetadata
from app.processing.crop import PlateRegion, crop_plate_region_from_array
from app.processing.resize import upscale_image_multiple

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """Manages the forensic image processing workflow."""
    
    def __init__(
        self,
        session_id: str,
        evidence_id: str,
        original_sha256: str,
        output_base_dir: str | Path = "output"
    ):
        """
        Initialize processing pipeline.
        
        Args:
            session_id: Session ID
            evidence_id: Evidence ID
            original_sha256: SHA-256 of original evidence
            output_base_dir: Base directory for output files
        """
        self.session_id = session_id
        self.evidence_id = evidence_id
        self.original_sha256 = original_sha256
        
        # Create session output directory
        self.output_dir = Path(output_base_dir) / session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.crops_dir = self.output_dir / "crops"
        self.enhancement_dir = self.output_dir / "enhancement"
        self.threshold_dir = self.output_dir / "threshold"
        self.deblur_dir = self.output_dir / "deblur"
        self.comparison_dir = self.output_dir / "comparison"
        self.metadata_dir = self.output_dir / "metadata"
        self.evidence_dir = self.output_dir / "evidence"
        
        for directory in [
            self.crops_dir, self.enhancement_dir, self.threshold_dir,
            self.deblur_dir, self.comparison_dir, self.metadata_dir,
            self.evidence_dir
        ]:
            directory.mkdir(exist_ok=True)
        
        # Initialize metadata manager
        self.metadata = ProcessingMetadata(
            session_id,
            evidence_id,
            original_sha256
        )
        
        # Store processing results
        self.results: Dict[str, np.ndarray] = {}
        self.file_hashes: Dict[str, str] = {}
        
        logger.info(f"Pipeline initialized for session {session_id}")
    
    def save_image(
        self,
        image: np.ndarray,
        filename: str,
        directory: Optional[Path] = None,
        operation_name: Optional[str] = None,
        operation_params: Optional[dict] = None
    ) -> Path:
        """
        Save an image file and record operation metadata.
        
        Args:
            image: Image as numpy array
            filename: Output filename
            directory: Output directory (default: enhancement_dir)
            operation_name: Name of operation that produced this image
            operation_params: Parameters used in the operation
            
        Returns:
            Path to saved file
        """
        if directory is None:
            directory = self.enhancement_dir
        
        output_path = directory / filename
        
        # Save image
        success = cv2.imwrite(str(output_path), image)
        if not success:
            raise IOError(f"Failed to save image: {output_path}")
        
        # Calculate hash
        _, buffer = cv2.imencode('.png', image)
        file_hash = calculate_data_sha256(buffer.tobytes())
        self.file_hashes[filename] = file_hash
        
        # Record in results
        self.results[filename] = image
        
        # Record operation metadata
        if operation_name:
            self.metadata.add_operation(
                operation_name=operation_name,
                parameters=operation_params or {},
                output_file=filename,
                output_hash=file_hash
            )
        
        logger.info(f"Image saved: {filename} ({file_hash[:16]}...)")
        
        return output_path
    
    def process_plate_crop(
        self,
        image: np.ndarray,
        region: PlateRegion
    ) -> np.ndarray:
        """
        Extract and process plate crop.
        
        Args:
            image: Original image
            region: Plate region to crop
            
        Returns:
            Cropped plate image
        """
        cropped = crop_plate_region_from_array(image, region)
        
        # Save original crop
        self.save_image(
            cropped,
            "01_original_crop.png",
            directory=self.crops_dir,
            operation_name="crop",
            operation_params=region.to_dict()
        )
        
        logger.info("Plate crop created and saved")
        
        return cropped
    
    def process_upscaling(
        self,
        image: np.ndarray,
        scale_factors: list[float] = [2.0, 4.0, 8.0],
        interpolation: str = "LANCZOS4"
    ) -> Dict[float, np.ndarray]:
        """
        Generate upscaled versions of image.
        
        Args:
            image: Input image
            scale_factors: Scale factors to generate (default: [2.0, 4.0, 8.0])
            interpolation: Interpolation method (default: LANCZOS4)
            
        Returns:
            Dictionary mapping scale factors to upscaled images
        """
        upscaled_images = upscale_image_multiple(
            image,
            scale_factors=scale_factors,
            interpolation=interpolation
        )
        
        filenames = {
            2.0: "02_upscale_2x.png",
            4.0: "03_upscale_4x.png",
            8.0: "04_upscale_8x.png"
        }
        
        for scale_factor, upscaled in upscaled_images.items():
            if scale_factor in filenames:
                filename = filenames[scale_factor]
                self.save_image(
                    upscaled,
                    filename,
                    directory=self.enhancement_dir,
                    operation_name="upscale",
                    operation_params={
                        "scale_factor": scale_factor,
                        "interpolation": interpolation
                    }
                )
        
        logger.info(f"Generated {len(upscaled_images)} upscaled images")
        
        return upscaled_images
    
    def save_evidence_metadata(self, evidence_data: dict) -> Path:
        """
        Save evidence metadata to JSON.
        
        Args:
            evidence_data: Evidence metadata dictionary
            
        Returns:
            Path to saved metadata file
        """
        import json
        
        metadata_file = self.evidence_dir / "original_reference.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(evidence_data, f, indent=2)
        
        logger.info(f"Evidence metadata saved: {metadata_file}")
        
        return metadata_file
    
    def save_processing_manifest(self) -> Path:
        """
        Save processing manifest JSON.
        
        Returns:
            Path to saved manifest
        """
        manifest_file = self.metadata_dir / "processing_manifest.json"
        self.metadata.save_manifest(manifest_file)
        
        logger.info(f"Processing manifest saved: {manifest_file}")
        
        return manifest_file
    
    def process_deblurring_pipeline(
        self,
        image: np.ndarray,
        enable_wiener: bool = True,
        enable_richardson_lucy: bool = True,
        enable_motion_blur: bool = True
    ) -> dict:
        """
        Apply experimental deblurring methods.
        
        WARNING: These are EXPERIMENTAL methods. Results should NOT be treated
        as recovered ground truth. Manual verification is required.
        
        Args:
            image: Input image
            enable_wiener: Whether to apply Wiener filter (default: True)
            enable_richardson_lucy: Whether to apply Richardson-Lucy (default: True)
            enable_motion_blur: Whether to apply motion blur removal (default: True)
            
        Returns:
            Dictionary with deblurred images
        """
        from app.processing.deblur import (
            apply_wiener_filter, apply_richardson_lucy_deconvolution,
            apply_motion_blur_removal
        )
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Wiener deblurring
        if enable_wiener:
            wiener = apply_wiener_filter(gray, noise_variance=10.0, kernel_size=5)
            self.save_image(
                wiener,
                "12_deblur_wiener.png",
                operation_name="wiener_deblurring",
                operation_params={"noise_variance": 10.0, "kernel_size": 5, "experimental": True}
            )
        
        # Apply Richardson-Lucy deconvolution
        if enable_richardson_lucy:
            rl = apply_richardson_lucy_deconvolution(
                gray,
                kernel_size=5,
                iterations=10,
                regularization=0.1
            )
            self.save_image(
                rl,
                "13_deblur_rl.png",
                operation_name="richardson_lucy_deconvolution",
                operation_params={"kernel_size": 5, "iterations": 10, "regularization": 0.1, "experimental": True}
            )
        
        # Apply motion blur removal
        if enable_motion_blur:
            motion = apply_motion_blur_removal(
                gray,
                kernel_size=11,
                angle=0.0,
                iterations=5
            )
            self.save_image(
                motion,
                "14_deblur_motion.png",
                operation_name="motion_blur_removal",
                operation_params={"kernel_size": 11, "angle": 0.0, "iterations": 5, "experimental": True}
            )
        
        logger.info("Deblurring pipeline completed (EXPERIMENTAL)")
        
        return self.results
    
    def get_output_summary(self) -> dict:
        """
        Get summary of pipeline output.
        
        Returns:
            Dictionary with output summary
        """
        return {
            "session_id": self.session_id,
            "output_directory": str(self.output_dir),
            "files_saved": len(self.results),
            "total_hashes": len(self.file_hashes),
            "operations_recorded": len(self.metadata.operations),
            "file_list": list(self.results.keys()),
            "file_hashes": self.file_hashes
        }
