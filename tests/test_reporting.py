"""Tests for reporting and manifest generation modules."""

import tempfile
from pathlib import Path
import unittest

from app.reporting.report import ForensicReportGenerator
from app.reporting.manifest import ProcessingManifestSchema, ManifestGenerator


class TestForensicReportGenerator(unittest.TestCase):
    """Test forensic report generation."""
    
    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        self.session_id = "SES-2026-TEST001"
        self.evidence_id = "EV-2026-TEST001"
        self.generator = ForensicReportGenerator(
            self.session_id,
            self.evidence_id,
            software_version="1.0.0"
        )
    
    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()
    
    def test_generate_header(self):
        """Test HTML header generation."""
        header = self.generator.generate_header()
        
        self.assertIn("<!DOCTYPE html>", header)
        self.assertIn(self.evidence_id, header)
        self.assertIn("Forensic Analysis Report", header)
    
    def test_generate_footer(self):
        """Test HTML footer generation."""
        footer = self.generator.generate_footer()
        
        self.assertIn("</html>", footer)
        self.assertIn(self.generator.software_version, footer)
    
    def test_generate_executive_summary(self):
        """Test executive summary section."""
        summary = self.generator.generate_executive_summary(
            "test_evidence.jpg",
            "abc123def456",
            5,
            3
        )
        
        self.assertIn("Executive Summary", summary)
        self.assertIn(self.session_id, summary)
        self.assertIn(self.evidence_id, summary)
        self.assertIn("test_evidence.jpg", summary)
        self.assertIn("abc123def456", summary)
    
    def test_generate_evidence_section(self):
        """Test evidence section generation."""
        evidence_data = {
            "filename": "test.jpg",
            "resolution": "640x480",
            "sha256": "abc123"
        }
        
        section = self.generator.generate_evidence_section(evidence_data)
        
        self.assertIn("Evidence Information", section)
        self.assertIn("test.jpg", section)
        self.assertIn("640x480", section)
    
    def test_generate_operations_section(self):
        """Test operations section generation."""
        operations = [
            {
                "operation": "upscale",
                "parameters": {"scale": 2},
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
        
        section = self.generator.generate_operations_section(operations)
        
        self.assertIn("Processing Operations", section)
        self.assertIn("upscale", section)
        self.assertIn("clahe", section)
    
    def test_generate_integrity_section(self):
        """Test integrity section generation."""
        derivatives = {
            "file1.png": "hash1abc",
            "file2.png": "hash2def"
        }
        
        section = self.generator.generate_integrity_section(
            "originalhash",
            derivatives
        )
        
        self.assertIn("Integrity Verification", section)
        self.assertIn("file1.png", section)
        self.assertIn("file2.png", section)
    
    def test_generate_limitations_section(self):
        """Test limitations section generation."""
        section = self.generator.generate_limitations_section()
        
        self.assertIn("Scientific and Forensic Limitations", section)
        self.assertIn("EXPERIMENTAL", section)
        self.assertIn("Manual verification", section)
    
    def test_generate_report(self):
        """Test complete report generation."""
        evidence_data = {
            "filename": "test.jpg",
            "sha256": "abc123",
            "resolution": "640x480"
        }
        
        operations = [
            {
                "operation": "upscale",
                "parameters": {"scale": 2},
                "output_file": "upscaled.png",
                "output_hash": "hash1"
            }
        ]
        
        derivatives_hashes = {
            "upscaled.png": "hash1",
            "enhanced.png": "hash2"
        }
        
        report_path = self.generator.generate_report(
            evidence_data,
            operations,
            derivatives_hashes,
            self.temp_path / "report.html"
        )
        
        self.assertTrue(report_path.exists())
        
        # Verify content
        with open(report_path, 'r') as f:
            content = f.read()
        
        self.assertIn("Forensic Analysis Report", content)
        self.assertIn(self.session_id, content)
        self.assertIn(self.evidence_id, content)


class TestProcessingManifestSchema(unittest.TestCase):
    """Test manifest schema."""
    
    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        self.session_id = "SES-2026-TEST001"
        self.evidence_id = "EV-2026-TEST001"
        self.original_sha256 = "abc123def456"
    
    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()
    
    def test_create_manifest(self):
        """Test manifest creation."""
        operations = [
            {
                "operation": "upscale",
                "parameters": {"scale": 2},
                "output_file": "upscaled.png",
                "output_hash": "hash1"
            }
        ]
        
        manifest = ProcessingManifestSchema.create_manifest(
            self.session_id,
            self.evidence_id,
            self.original_sha256,
            operations
        )
        
        self.assertIn("forensic_manifest", manifest)
        manifest_data = manifest["forensic_manifest"]
        
        self.assertEqual(manifest_data["session"]["session_id"], self.session_id)
        self.assertEqual(manifest_data["evidence"]["evidence_id"], self.evidence_id)
        self.assertEqual(manifest_data["evidence"]["original_sha256"], self.original_sha256)
    
    def test_save_and_load_manifest(self):
        """Test saving and loading manifest."""
        operations = []
        
        manifest = ProcessingManifestSchema.create_manifest(
            self.session_id,
            self.evidence_id,
            self.original_sha256,
            operations
        )
        
        manifest_path = ProcessingManifestSchema.save_manifest(
            manifest,
            self.temp_path / "manifest.json"
        )
        
        self.assertTrue(manifest_path.exists())
        
        # Load and verify
        loaded = ProcessingManifestSchema.load_manifest(manifest_path)
        
        self.assertIn("forensic_manifest", loaded)
    
    def test_validate_manifest(self):
        """Test manifest validation."""
        operations = []
        
        manifest = ProcessingManifestSchema.create_manifest(
            self.session_id,
            self.evidence_id,
            self.original_sha256,
            operations
        )
        
        is_valid = ProcessingManifestSchema.validate_manifest(manifest)
        
        self.assertTrue(is_valid)
    
    def test_validate_invalid_manifest(self):
        """Test validation of invalid manifest."""
        invalid_manifest = {}
        
        is_valid = ProcessingManifestSchema.validate_manifest(invalid_manifest)
        
        self.assertFalse(is_valid)


class TestManifestGenerator(unittest.TestCase):
    """Test manifest generator utility."""
    
    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        self.session_id = "SES-2026-TEST001"
        self.evidence_id = "EV-2026-TEST001"
        self.original_sha256 = "abc123"
        
        self.generator = ManifestGenerator(
            self.session_id,
            self.evidence_id,
            self.original_sha256
        )
    
    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()
    
    def test_add_operation(self):
        """Test adding operations."""
        self.generator.add_operation(
            "upscale",
            parameters={"scale": 2},
            output_file="upscaled.png",
            output_hash="hash1"
        )
        
        self.assertEqual(self.generator.get_operations_count(), 1)
        self.assertEqual(self.generator.get_derivatives_count(), 1)
    
    def test_add_multiple_operations(self):
        """Test adding multiple operations."""
        self.generator.add_operation(
            "upscale",
            parameters={"scale": 2},
            output_file="upscaled.png",
            output_hash="hash1"
        )
        
        self.generator.add_operation(
            "clahe",
            parameters={"clip_limit": 2.0},
            output_file="enhanced.png",
            output_hash="hash2"
        )
        
        self.assertEqual(self.generator.get_operations_count(), 2)
        self.assertEqual(self.generator.get_derivatives_count(), 2)
    
    def test_generate_manifest(self):
        """Test manifest generation."""
        self.generator.add_operation(
            "upscale",
            parameters={"scale": 2},
            output_file="upscaled.png",
            output_hash="hash1"
        )
        
        manifest = self.generator.generate_manifest()
        
        self.assertIn("forensic_manifest", manifest)
    
    def test_save_manifest(self):
        """Test saving generated manifest."""
        self.generator.add_operation(
            "test_op",
            output_file="test.png",
            output_hash="testhash"
        )
        
        manifest_path = self.generator.save_manifest(
            self.temp_path / "manifest.json"
        )
        
        self.assertTrue(manifest_path.exists())
    
    def test_get_operations_summary(self):
        """Test operations summary."""
        self.generator.add_operation(
            "upscale",
            output_file="upscaled.png",
            output_hash="hash1"
        )
        
        summary = self.generator.get_operations_summary()
        
        self.assertIn(self.session_id, summary)
        self.assertIn(self.evidence_id, summary)
        self.assertIn("1", summary)  # Operations count


if __name__ == "__main__":
    unittest.main()
