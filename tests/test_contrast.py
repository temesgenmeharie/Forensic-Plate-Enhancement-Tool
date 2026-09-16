"""Tests for contrast enhancement module."""

import unittest

import numpy as np

from app.processing.contrast import (
    apply_clahe, apply_clahe_color, apply_histogram_equalization,
    calculate_contrast_metrics
)


class TestCLAHE(unittest.TestCase):
    """Test CLAHE enhancement."""
    
    def setUp(self):
        """Create test images."""
        self.gray_image = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
        self.color_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    
    def test_apply_clahe_grayscale(self):
        """Test CLAHE on grayscale image."""
        enhanced = apply_clahe(self.gray_image)
        
        self.assertEqual(enhanced.shape, self.gray_image.shape)
        self.assertTrue(np.all(enhanced >= 0))
        self.assertTrue(np.all(enhanced <= 255))
    
    def test_apply_clahe_color(self):
        """Test CLAHE on color image."""
        enhanced = apply_clahe(self.color_image)
        
        # Should return grayscale
        self.assertEqual(len(enhanced.shape), 2)
    
    def test_apply_clahe_custom_parameters(self):
        """Test CLAHE with custom parameters."""
        enhanced = apply_clahe(
            self.gray_image,
            clip_limit=4.0,
            tile_grid_size=(16, 16)
        )
        
        self.assertEqual(enhanced.shape, self.gray_image.shape)
    
    def test_apply_clahe_invalid_clip_limit(self):
        """Test CLAHE with invalid clip limit."""
        with self.assertRaises(ValueError):
            apply_clahe(self.gray_image, clip_limit=0)
        
        with self.assertRaises(ValueError):
            apply_clahe(self.gray_image, clip_limit=-1)
    
    def test_apply_clahe_invalid_tile_grid(self):
        """Test CLAHE with invalid tile grid."""
        with self.assertRaises(ValueError):
            apply_clahe(self.gray_image, tile_grid_size=(0, 8))
        
        with self.assertRaises(ValueError):
            apply_clahe(self.gray_image, tile_grid_size=(8, -1))


class TestCLAHEColor(unittest.TestCase):
    """Test color-preserving CLAHE."""
    
    def setUp(self):
        """Create test color image."""
        self.color_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    
    def test_apply_clahe_color_preserving(self):
        """Test CLAHE with color preservation."""
        enhanced = apply_clahe_color(self.color_image)
        
        self.assertEqual(enhanced.shape, self.color_image.shape)
        self.assertEqual(enhanced.shape[2], 3)
    
    def test_apply_clahe_color_invalid_input(self):
        """Test CLAHE with invalid color input."""
        gray_image = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        
        with self.assertRaises(ValueError):
            apply_clahe_color(gray_image)


class TestHistogramEqualization(unittest.TestCase):
    """Test histogram equalization."""
    
    def setUp(self):
        """Create test image."""
        self.gray_image = np.random.randint(50, 150, (480, 640), dtype=np.uint8)
    
    def test_histogram_equalization(self):
        """Test histogram equalization."""
        equalized = apply_histogram_equalization(self.gray_image)
        
        self.assertEqual(equalized.shape, self.gray_image.shape)
        self.assertTrue(np.all(equalized >= 0))
        self.assertTrue(np.all(equalized <= 255))


class TestContrastMetrics(unittest.TestCase):
    """Test contrast metrics calculation."""
    
    def setUp(self):
        """Create test images."""
        self.gray_image = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
        self.color_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    
    def test_calculate_contrast_metrics_grayscale(self):
        """Test calculating metrics for grayscale image."""
        metrics = calculate_contrast_metrics(self.gray_image)
        
        self.assertIn("mean", metrics)
        self.assertIn("std_dev", metrics)
        self.assertIn("min", metrics)
        self.assertIn("max", metrics)
        self.assertIn("dynamic_range", metrics)
    
    def test_calculate_contrast_metrics_color(self):
        """Test calculating metrics for color image."""
        metrics = calculate_contrast_metrics(self.color_image)
        
        self.assertIn("mean", metrics)
        self.assertGreater(metrics["max"], metrics["min"])


if __name__ == "__main__":
    unittest.main()
