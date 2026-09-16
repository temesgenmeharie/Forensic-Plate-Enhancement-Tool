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
    
    def process_enhancement_pipeline(
        self,
        image: np.ndarray,
        apply_clahe: bool = True,
        apply_denoising: bool = True,
        apply_sharpening: bool = True,
        apply_thresholding: bool = True
    ) -> dict:
        """
        Apply full enhancement pipeline to image.
        
        Args:
            image: Input image
            apply_clahe: Whether to apply CLAHE (default: True)
            apply_denoising: Whether to apply denoising (default: True)
            apply_sharpening: Whether to apply sharpening (default: True)
            apply_thresholding: Whether to apply thresholding (default: True)
            
        Returns:
            Dictionary with all processed images
        """
        from app.processing.contrast import apply_clahe
        from app.processing.denoise import apply_bilateral_filter, apply_non_local_means_denoise
        from app.processing.sharpen import apply_unsharp_mask
        from app.processing.threshold import apply_adaptive_gaussian_threshold, apply_adaptive_mean_threshold
        
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        current = gray.copy()
        
        # Apply CLAHE
        if apply_clahe:
            enhanced_clahe = apply_clahe(current, clip_limit=2.0, tile_grid_size=(8, 8))
            self.save_image(
                enhanced_clahe,
                "06_clahe.png",
                operation_name="clahe",
                operation_params={"clip_limit": 2.0, "tile_grid_size": [8, 8]}
            )
            current = enhanced_clahe
        
        # Apply bilateral denoising
        if apply_denoising:
            denoised_bilateral = apply_bilateral_filter(current, diameter=9, sigma_color=75, sigma_space=75)
            self.save_image(
                denoised_bilateral,
                "07_bilateral.png",
                operation_name="bilateral_denoise",
                operation_params={"diameter": 9, "sigma_color": 75, "sigma_space": 75}
            )
        
        # Apply NLM denoising
        if apply_denoising:
            denoised_nlm = apply_non_local_means_denoise(current, h=10, template_window_size=7, search_window_size=21)
            self.save_image(
                denoised_nlm,
                "08_nlm.png",
                operation_name="nlm_denoise",
                operation_params={"h": 10, "template_window_size": 7, "search_window_size": 21}
            )
            current = denoised_nlm
        
        # Apply sharpening
        if apply_sharpening:
            sharpened = apply_unsharp_mask(current, sigma=1.0, amount=1.5, threshold=0)
            self.save_image(
                sharpened,
                "09_unsharp.png",
                operation_name="unsharp_mask",
                operation_params={"sigma": 1.0, "amount": 1.5, "threshold": 0}
            )
        
        # Apply adaptive thresholding
        if apply_thresholding:
            threshold_gaussian = apply_adaptive_gaussian_threshold(current, block_size=11, constant=2.0)
            self.save_image(
                threshold_gaussian,
                "10_adaptive_gaussian.png",
                operation_name="adaptive_gaussian_threshold",
                operation_params={"block_size": 11, "constant": 2.0}
            )
            
            threshold_mean = apply_adaptive_mean_threshold(current, block_size=11, constant=2.0)
            self.save_image(
                threshold_mean,
                "11_adaptive_mean.png",
                operation_name="adaptive_mean_threshold",
                operation_params={"block_size": 11, "constant": 2.0}
            )
        
        logger.info("Enhancement pipeline completed")
        
        return self.results
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
