"""Tests for processing pipeline."""

import tempfile
from pathlib import Path
import unittest

import numpy as np

from app.processing.pipeline import ProcessingPipeline
from app.processing.crop import PlateRegion


class TestProcessingPipeline(unittest.TestCase):
    """Test processing pipeline functionality."""
    
    def setUp(self):
        """Create temporary directory and test image."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        self.test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        
        self.session_id = "SES-2026-TEST001"
        self.evidence_id = "EV-TEST-001"
        self.original_sha256 = "a" * 64
    
    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = ProcessingPipeline(
            self.session_id,
            self.evidence_id,
            self.original_sha256,
            output_base_dir=self.temp_path
        )
        
        self.assertEqual(pipeline.session_id, self.session_id)
        self.assertEqual(pipeline.evidence_id, self.evidence_id)
        
        # Check that directories were created
        self.assertTrue(pipeline.crops_dir.exists())
        self.assertTrue(pipeline.enhancement_dir.exists())
    
    def test_save_image(self):
        """Test saving image through pipeline."""
        pipeline = ProcessingPipeline(
            self.session_id,
            self.evidence_id,
            self.original_sha256,
            output_base_dir=self.temp_path
        )
        
        output_path = pipeline.save_image(
            self.test_image,
            "test_image.png",
            operation_name="test"
        )
        
        self.assertTrue(output_path.exists())
        self.assertIn("test_image.png", pipeline.results)
        self.assertIn("test_image.png", pipeline.file_hashes)
    
    def test_process_plate_crop(self):
        """Test plate crop processing."""
        pipeline = ProcessingPipeline(
            self.session_id,
            self.evidence_id,
            self.original_sha256,
            output_base_dir=self.temp_path
        )
        
        region = PlateRegion(100, 150, 300, 100)
        cropped = pipeline.process_plate_crop(self.test_image, region)
        
        self.assertEqual(cropped.shape[0], 100)
        self.assertEqual(cropped.shape[1], 300)
        
        # Check that crop was saved
        crop_file = pipeline.crops_dir / "01_original_crop.png"
        self.assertTrue(crop_file.exists())
    
    def test_process_upscaling(self):
        """Test upscaling processing."""
        pipeline = ProcessingPipeline(
            self.session_id,
            self.evidence_id,
            self.original_sha256,
            output_base_dir=self.temp_path
        )
        
        # Use smaller test image for efficiency
        small_image = self.test_image[100:200, 100:300]
        
        upscaled = pipeline.process_upscaling(
            small_image,
            scale_factors=[2.0, 4.0],
            interpolation="LANCZOS4"
        )
        
        self.assertEqual(len(upscaled), 2)
        
        # Check that files were saved
        self.assertTrue((pipeline.enhancement_dir / "02_upscale_2x.png").exists())
        self.assertTrue((pipeline.enhancement_dir / "03_upscale_4x.png").exists())
    
    def test_save_evidence_metadata(self):
        """Test saving evidence metadata."""
        pipeline = ProcessingPipeline(
            self.session_id,
            self.evidence_id,
            self.original_sha256,
            output_base_dir=self.temp_path
        )
        
        evidence_data = {
            "evidence_id": self.evidence_id,
            "sha256": self.original_sha256,
            "type": "image"
        }
        
        metadata_path = pipeline.save_evidence_metadata(evidence_data)
        
        self.assertTrue(metadata_path.exists())
    
    def test_save_processing_manifest(self):
        """Test saving processing manifest."""
        pipeline = ProcessingPipeline(
            self.session_id,
            self.evidence_id,
            self.original_sha256,
            output_base_dir=self.temp_path
        )
        
        manifest_path = pipeline.save_processing_manifest()
        
        self.assertTrue(manifest_path.exists())
    
    def test_get_output_summary(self):
        """Test getting output summary."""
        pipeline = ProcessingPipeline(
            self.session_id,
            self.evidence_id,
            self.original_sha256,
            output_base_dir=self.temp_path
        )
        
        # Save an image
        pipeline.save_image(self.test_image, "test.png")
        
        summary = pipeline.get_output_summary()
        
        self.assertIn("session_id", summary)
        self.assertIn("output_directory", summary)
        self.assertIn("files_saved", summary)
        self.assertGreater(summary["files_saved"], 0)


if __name__ == "__main__":
    unittest.main()
