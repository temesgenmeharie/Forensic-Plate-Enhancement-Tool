"""Tests for evidence management module."""

import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from app.core.evidence import Evidence, SUPPORTED_IMAGE_FORMATS, SUPPORTED_VIDEO_FORMATS


class TestEvidence(unittest.TestCase):
    """Test Evidence functionality."""
    
    def setUp(self):
        """Create temporary test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def _create_test_image(self, filename: str = "test.jpg") -> Path:
        """Create a test image file."""
        img_path = self.temp_path / filename
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        return img_path
    
    def test_evidence_creation_image(self):
        """Test creating an evidence record for an image."""
        img_path = self._create_test_image()
        evidence = Evidence(img_path)
        
        self.assertIsNotNone(evidence.evidence_id)
        self.assertTrue(evidence.evidence_id.startswith("EV-"))
        self.assertEqual(evidence.evidence_type, "image")
        self.assertEqual(evidence.original_filename, "test.jpg")
    
    def test_evidence_sha256_calculated(self):
        """Test that SHA-256 is calculated for evidence."""
        img_path = self._create_test_image()
        evidence = Evidence(img_path)
        
        self.assertIsNotNone(evidence.sha256)
        self.assertEqual(len(evidence.sha256), 64)
    
    def test_evidence_image_metadata(self):
        """Test that image metadata is loaded."""
        img_path = self._create_test_image()
        evidence = Evidence(img_path)
        
        self.assertEqual(evidence.width, 640)
        self.assertEqual(evidence.height, 480)
        self.assertEqual(evidence.resolution, "640x480")
        self.assertEqual(evidence.channels, 3)
    
    def test_evidence_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with self.assertRaises(FileNotFoundError):
            Evidence(self.temp_path / "nonexistent.jpg")
    
    def test_evidence_unsupported_format(self):
        """Test that ValueError is raised for unsupported formats."""
        unsupported_file = self.temp_path / "test.xyz"
        unsupported_file.write_text("test")
        
        with self.assertRaises(ValueError):
            Evidence(unsupported_file)
    
    def test_evidence_to_dict(self):
        """Test exporting evidence as dictionary."""
        img_path = self._create_test_image()
        evidence = Evidence(img_path)
        
        data = evidence.to_dict()
        
        self.assertIn("evidence_id", data)
        self.assertIn("original_filename", data)
        self.assertIn("sha256", data)
        self.assertIn("type", data)
        self.assertIn("resolution", data)
    
    def test_supported_formats(self):
        """Test that supported formats are defined."""
        self.assertGreater(len(SUPPORTED_IMAGE_FORMATS), 0)
        self.assertGreater(len(SUPPORTED_VIDEO_FORMATS), 0)
        self.assertIn(".jpg", SUPPORTED_IMAGE_FORMATS)
        self.assertIn(".png", SUPPORTED_IMAGE_FORMATS)
        self.assertIn(".mp4", SUPPORTED_VIDEO_FORMATS)


if __name__ == "__main__":
    unittest.main()
