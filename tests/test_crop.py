"""Tests for crop module."""

import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from app.processing.crop import PlateRegion, crop_plate_region, crop_plate_region_from_array, validate_region


class TestPlateRegion(unittest.TestCase):
    """Test PlateRegion functionality."""
    
    def test_plate_region_creation(self):
        """Test creating a plate region."""
        region = PlateRegion(10, 20, 300, 100)
        
        self.assertEqual(region.x, 10)
        self.assertEqual(region.y, 20)
        self.assertEqual(region.width, 300)
        self.assertEqual(region.height, 100)
    
    def test_plate_region_negative_coords(self):
        """Test that negative coordinates are clamped to 0."""
        region = PlateRegion(-10, -20, 300, 100)
        
        self.assertEqual(region.x, 0)
        self.assertEqual(region.y, 0)
    
    def test_plate_region_get_coordinates(self):
        """Test getting region coordinates."""
        region = PlateRegion(10, 20, 300, 100)
        coords = region.get_coordinates()
        
        self.assertEqual(coords, (10, 20, 300, 100))
    
    def test_plate_region_get_bbox(self):
        """Test getting bounding box."""
        region = PlateRegion(10, 20, 300, 100)
        bbox = region.get_bbox()
        
        self.assertEqual(bbox, (10, 20, 310, 120))
    
    def test_plate_region_to_dict(self):
        """Test exporting region as dictionary."""
        region = PlateRegion(10, 20, 300, 100)
        data = region.to_dict()
        
        self.assertEqual(data["x"], 10)
        self.assertEqual(data["y"], 20)
        self.assertEqual(data["width"], 300)
        self.assertEqual(data["height"], 100)
    
    def test_plate_region_from_dict(self):
        """Test creating region from dictionary."""
        data = {"x": 10, "y": 20, "width": 300, "height": 100}
        region = PlateRegion.from_dict(data)
        
        self.assertEqual(region.x, 10)
        self.assertEqual(region.y, 20)
        self.assertEqual(region.width, 300)
        self.assertEqual(region.height, 100)


class TestCropping(unittest.TestCase):
    """Test cropping functionality."""
    
    def setUp(self):
        """Create temporary test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def _create_test_image(self, width: int = 640, height: int = 480) -> Path:
        """Create a test image file."""
        img_path = self.temp_path / "test.jpg"
        img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        return img_path
    
    def test_crop_plate_region(self):
        """Test cropping a plate region from an image."""
        img_path = self._create_test_image()
        region = PlateRegion(100, 150, 300, 100)
        
        cropped = crop_plate_region(img_path, region)
        
        self.assertEqual(cropped.shape[0], 100)  # height
        self.assertEqual(cropped.shape[1], 300)  # width
    
    def test_crop_plate_region_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        region = PlateRegion(0, 0, 100, 100)
        
        with self.assertRaises(FileNotFoundError):
            crop_plate_region(self.temp_path / "nonexistent.jpg", region)
    
    def test_crop_plate_region_from_array(self):
        """Test cropping from image array."""
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        region = PlateRegion(100, 150, 300, 100)
        
        cropped = crop_plate_region_from_array(img, region)
        
        self.assertEqual(cropped.shape[0], 100)
        self.assertEqual(cropped.shape[1], 300)
    
    def test_crop_plate_region_clamping(self):
        """Test that region coordinates are clamped to image bounds."""
        img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        # Region extends beyond image
        region = PlateRegion(500, 400, 300, 200)
        
        cropped = crop_plate_region_from_array(img, region)
        
        # Should still produce valid output
        self.assertGreater(cropped.shape[0], 0)
        self.assertGreater(cropped.shape[1], 0)
    
    def test_validate_region_valid(self):
        """Test validating a valid region."""
        region = PlateRegion(100, 150, 300, 100)
        is_valid = validate_region(region, 640, 480)
        
        self.assertTrue(is_valid)
    
    def test_validate_region_out_of_bounds(self):
        """Test validating a region outside image bounds."""
        region = PlateRegion(500, 400, 300, 200)
        is_valid = validate_region(region, 640, 480)
        
        self.assertFalse(is_valid)
    
    def test_validate_region_invalid_size(self):
        """Test validating a region with invalid size."""
        region = PlateRegion(100, 150, 0, 0)
        is_valid = validate_region(region, 640, 480)
        
        self.assertFalse(is_valid)


if __name__ == "__main__":
    unittest.main()
