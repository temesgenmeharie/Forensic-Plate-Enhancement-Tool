"""Integration tests for complete end-to-end workflows."""

import unittest
import tempfile
from pathlib import Path
import cv2
import numpy as np

from app.core.evidence import Evidence
from app.core.session import Session
from app.processing.pipeline import ProcessingPipeline
from app.processing.resize import upscale_image_multiple
from app.processing.contrast import apply_clahe
from app.processing.denoise import apply_bilateral_filter, apply_non_local_means_denoise
from app.processing.sharpen import apply_unsharp_mask
from app.processing.threshold import apply_adaptive_gaussian_threshold
from app.reporting.report import ForensicReportGenerator
from app.reporting.manifest import ManifestGenerator


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete end-to-end forensic analysis workflow."""
    
    def setUp(self):
        """Create test image and temporary directory."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create a test image (license plate-like)
        self.test_image = np.ones((100, 200, 3), dtype=np.uint8) * 100
        cv2.putText(
            self.test_image,
            "ABC123",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            2
        )
        
        # Save test image
        self.test_image_path = self.temp_path / "test_plate.jpg"
        cv2.imwrite(str(self.test_image_path), self.test_image)
    
    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()
    
    def test_complete_enhancement_workflow(self):
        """Test complete image enhancement workflow."""
        # Load evidence
        evidence = Evidence(str(self.test_image_path))
        session = Session()
        
        # Verify evidence loaded
        self.assertEqual(evidence.evidence_type, "image")
        self.assertEqual(evidence.width, 200)
        self.assertEqual(evidence.height, 100)
        self.assertIsNotNone(evidence.sha256)
        
        # Create pipeline
        pipeline = ProcessingPipeline(
            session_id=session.session_id,
            evidence_id=evidence.evidence_id,
            original_sha256=evidence.sha256,
            output_base_dir=str(self.temp_path)
        )
        
        # Load image
        image = cv2.imread(str(self.test_image_path))
        
        # Apply processing pipeline
        # 1. Upscale 2x
        upscaled = upscale_image_multiple(
            image,
            scale_factors=[2.0],
            interpolation="LANCZOS4"
        )
        image = upscaled[2.0]
        
        # 2. Denoise
        image = apply_bilateral_filter(image)
        
        # 3. Enhance contrast
        image = apply_clahe(image, clip_limit=2.0)
        
        # 4. Sharpen
        image = apply_unsharp_mask(image)
        
        # 5. Threshold
        image = apply_adaptive_gaussian_threshold(image)
        
        # Save final result
        output_path = pipeline.save_image(
            image,
            "final_enhanced.png",
            operation_name="complete_enhancement",
            operation_params={
                "upscale": 2.0,
                "denoise": "bilateral",
                "clahe_clip": 2.0,
                "sharpen": "unsharp_mask",
                "threshold": "adaptive_gaussian"
            }
        )
        
        # Verify output
        self.assertTrue(output_path.exists())
        self.assertIn("final_enhanced.png", pipeline.file_hashes)
        
        # Test pipeline summary
        summary = pipeline.get_output_summary()
        self.assertGreater(summary["files_saved"], 0)
        self.assertIsNotNone(summary["file_hashes"])
    
    def test_manifest_generation_workflow(self):
        """Test processing manifest generation workflow."""
        evidence = Evidence(str(self.test_image_path))
        session = Session()
        
        # Create manifest generator
        manifest_gen = ManifestGenerator(
            session_id=session.session_id,
            evidence_id=evidence.evidence_id,
            original_sha256=evidence.sha256
        )
        
        # Add operations
        manifest_gen.add_operation(
            "upscale",
            parameters={"scale": 2.0, "interpolation": "LANCZOS4"},
            output_file="upscaled_2x.png",
            output_hash="hash1abc"
        )
        
        manifest_gen.add_operation(
            "denoise",
            parameters={"method": "bilateral"},
            output_file="denoised.png",
            output_hash="hash2def"
        )
        
        # Save manifest
        manifest_path = manifest_gen.save_manifest(
            self.temp_path / "manifest.json"
        )
        
        # Verify manifest
        self.assertTrue(manifest_path.exists())
        self.assertEqual(manifest_gen.get_operations_count(), 2)
        self.assertEqual(manifest_gen.get_derivatives_count(), 2)
        
        # Verify manifest content
        import json
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        self.assertIn("forensic_manifest", manifest)
        self.assertEqual(
            len(manifest["forensic_manifest"]["processing"]["operations"]),
            2
        )
    
    def test_report_generation_workflow(self):
        """Test forensic report generation workflow."""
        evidence = Evidence(str(self.test_image_path))
        session = Session()
        
        # Create report generator
        report_gen = ForensicReportGenerator(
            session_id=session.session_id,
            evidence_id=evidence.evidence_id,
            software_version="1.0.0"
        )
        
        # Prepare data
        evidence_data = evidence.to_dict()
        
        operations = [
            {
                "operation": "upscale",
                "parameters": {"scale": 2.0},
                "output_file": "upscaled.png",
                "output_hash": "hash1"
            },
            {
                "operation": "clahe",
                "parameters": {"clip_limit": 2.0},
                "output_file": "enhanced.png",
                "output_hash": "hash2"
            }
        ]
        
        derivatives_hashes = {
            "upscaled.png": "hash1abc123",
            "enhanced.png": "hash2def456"
        }
        
        # Generate report
        report_path = report_gen.generate_report(
            evidence_data=evidence_data,
            processing_operations=operations,
            derivatives_hashes=derivatives_hashes,
            output_path=self.temp_path / "report.html"
        )
        
        # Verify report
        self.assertTrue(report_path.exists())
        
        # Check content
        with open(report_path, 'r') as f:
            content = f.read()
        
        self.assertIn("Forensic Analysis Report", content)
        self.assertIn(session.session_id, content)
        self.assertIn("SHA-256", content)
        self.assertIn("Scientific and Forensic Limitations", content)
    
    def test_complete_workflow_with_all_components(self):
        """Test complete workflow: evidence → pipeline → manifest → report."""
        # Step 1: Load evidence
        evidence = Evidence(str(self.test_image_path))
        session = Session()
        
        # Step 2: Create and execute pipeline
        pipeline = ProcessingPipeline(
            session_id=session.session_id,
            evidence_id=evidence.evidence_id,
            original_sha256=evidence.sha256,
            output_base_dir=str(self.temp_path)
        )
        
        image = cv2.imread(str(self.test_image_path))
        
        # Process image
        upscaled = upscale_image_multiple(image, scale_factors=[2.0])
        image = upscaled[2.0]
        image = apply_bilateral_filter(image)
        image = apply_clahe(image, clip_limit=2.0)
        
        # Save
        pipeline.save_image(
            image,
            "processed.png",
            operation_name="enhancement",
            operation_params={"steps": 3}
        )
        
        # Step 3: Generate manifest
        manifest_gen = ManifestGenerator(
            session.session_id,
            evidence.evidence_id,
            evidence.sha256
        )
        
        manifest_gen.add_operation(
            "upscale",
            parameters={"scale": 2.0},
            output_file="processed.png",
            output_hash=pipeline.file_hashes.get("processed.png")
        )
        
        manifest_path = manifest_gen.save_manifest(
            self.temp_path / "manifest.json"
        )
        
        # Step 4: Generate report
        report_gen = ForensicReportGenerator(
            session.session_id,
            evidence.evidence_id
        )
        
        report_path = report_gen.generate_report(
            evidence_data=evidence.to_dict(),
            processing_operations=[
                {
                    "operation": "upscale",
                    "parameters": {"scale": 2.0},
                    "output_file": "processed.png",
                    "output_hash": pipeline.file_hashes.get("processed.png")
                }
            ],
            derivatives_hashes=pipeline.file_hashes,
            output_path=self.temp_path / "report.html"
        )
        
        # Verify all outputs exist
        self.assertTrue(manifest_path.exists())
        self.assertTrue(report_path.exists())
        
        # Verify manifest
        import json
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        self.assertIn("forensic_manifest", manifest)
        
        # Verify report
        with open(report_path, 'r') as f:
            report = f.read()
        self.assertIn("Forensic Analysis Report", report)
        self.assertIn(evidence.evidence_id, report)


class TestWorkflowErrorHandling(unittest.TestCase):
    """Test error handling in workflows."""
    
    def setUp(self):
        """Create test directory."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()
    
    def test_missing_evidence_file(self):
        """Test handling of missing evidence file."""
        with self.assertRaises(FileNotFoundError):
            Evidence(str(self.temp_path / "nonexistent.jpg"))
    
    def test_unsupported_file_format(self):
        """Test handling of unsupported file format."""
        # Create an unsupported file
        unsupported_path = self.temp_path / "test.txt"
        unsupported_path.write_text("not an image")
        
        with self.assertRaises(ValueError):
            Evidence(str(unsupported_path))
    
    def test_pipeline_with_invalid_parameters(self):
        """Test pipeline behavior with edge case parameters."""
        # Create minimal test image
        test_image = np.ones((50, 50, 3), dtype=np.uint8) * 100
        test_path = self.temp_path / "test.jpg"
        cv2.imwrite(str(test_path), test_image)
        
        evidence = Evidence(str(test_path))
        
        # Test with minimum parameters
        pipeline = ProcessingPipeline(
            session_id="TEST-2026-000001",
            evidence_id=evidence.evidence_id,
            original_sha256=evidence.sha256,
            output_base_dir=str(self.temp_path)
        )
        
        # Should not raise exception
        self.assertIsNotNone(pipeline)


if __name__ == "__main__":
    unittest.main()
