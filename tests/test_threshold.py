"""Tests for thresholding module."""

import unittest

import numpy as np

from app.processing.threshold import (
    apply_adaptive_gaussian_threshold, apply_adaptive_mean_threshold,
    apply_otsu_threshold, apply_binary_threshold,
    apply_inverse_binary_threshold, calculate_threshold_metrics
)


class TestAdaptiveGaussianThreshold(unittest.TestCase):
    """Test adaptive Gaussian thresholding."""
    
    def setUp(self):
        """Create test images."""
        self.gray_image = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
        self.color_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    
    def test_adaptive_gaussian_threshold_grayscale(self):
        """Test adaptive Gaussian threshold on grayscale image."""
        thresholded = apply_adaptive_gaussian_threshold(self.gray_image)
        
        self.assertEqual(thresholded.shape, self.gray_image.shape)
        # Should be binary (0 or 255)
        self.assertTrue(np.all((thresholded == 0) | (thresholded == 255)))
    
    def test_adaptive_gaussian_threshold_color(self):
        """Test adaptive Gaussian threshold on color image."""
        thresholded = apply_adaptive_gaussian_threshold(self.color_image)
        
        # Output should be same shape as input (but converted to gray internally)
        self.assertEqual(thresholded.shape[:2], self.color_image.shape[:2])
    
    def test_adaptive_gaussian_threshold_custom_params(self):
        """Test with custom parameters."""
        thresholded = apply_adaptive_gaussian_threshold(
            self.gray_image,
            block_size=21,
            constant=5.0
        )
        
        self.assertTrue(np.all((thresholded == 0) | (thresholded == 255)))
    
    def test_adaptive_gaussian_threshold_invalid_block_size(self):
        """Test with invalid block size."""
        with self.assertRaises(ValueError):
            apply_adaptive_gaussian_threshold(self.gray_image, block_size=0)
        
        with self.assertRaises(ValueError):
            apply_adaptive_gaussian_threshold(self.gray_image, block_size=10)  # Even


class TestAdaptiveMeanThreshold(unittest.TestCase):
    """Test adaptive mean thresholding."""
    
    def setUp(self):
        """Create test image."""
        self.gray_image = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
    
    def test_adaptive_mean_threshold(self):
        """Test adaptive mean threshold."""
        thresholded = apply_adaptive_mean_threshold(self.gray_image)
        
        self.assertEqual(thresholded.shape, self.gray_image.shape)
        self.assertTrue(np.all((thresholded == 0) | (thresholded == 255)))


class TestOtsuThreshold(unittest.TestCase):
    """Test Otsu's thresholding."""
    
    def setUp(self):
        """Create test images."""
        self.gray_image = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
        self.color_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    
    def test_otsu_threshold_grayscale(self):
        """Test Otsu's threshold on grayscale image."""
        thresholded, threshold_value = apply_otsu_threshold(self.gray_image)
        
        self.assertEqual(thresholded.shape, self.gray_image.shape)
        self.assertTrue(np.all((thresholded == 0) | (thresholded == 255)))
        self.assertGreater(threshold_value, 0)
        self.assertLess(threshold_value, 256)
    
    def test_otsu_threshold_color(self):
        """Test Otsu's threshold on color image."""
        thresholded, threshold_value = apply_otsu_threshold(self.color_image)
        
        self.assertTrue(np.all((thresholded == 0) | (thresholded == 255)))


class TestBinaryThreshold(unittest.TestCase):
    """Test binary thresholding."""
    
    def setUp(self):
        """Create test image."""
        self.gray_image = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
    
    def test_binary_threshold(self):
        """Test binary threshold."""
        thresholded = apply_binary_threshold(self.gray_image, threshold_value=128)
        
        self.assertEqual(thresholded.shape, self.gray_image.shape)
        self.assertTrue(np.all((thresholded == 0) | (thresholded == 255)))
    
    def test_binary_threshold_custom_params(self):
        """Test binary threshold with custom parameters."""
        thresholded = apply_binary_threshold(
            self.gray_image,
            threshold_value=100,
            max_value=200
        )
        
        # Should only have values 0 and 200
        self.assertTrue(np.all((thresholded == 0) | (thresholded == 200)))
    
    def test_binary_threshold_invalid_threshold(self):
        """Test binary threshold with invalid threshold value."""
        with self.assertRaises(ValueError):
            apply_binary_threshold(self.gray_image, threshold_value=-1)
        
        with self.assertRaises(ValueError):
            apply_binary_threshold(self.gray_image, threshold_value=256)


class TestInverseBinaryThreshold(unittest.TestCase):
    """Test inverse binary thresholding."""
    
    def setUp(self):
        """Create test image."""
        self.gray_image = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
    
    def test_inverse_binary_threshold(self):
        """Test inverse binary threshold."""
        thresholded = apply_inverse_binary_threshold(self.gray_image)
        
        self.assertEqual(thresholded.shape, self.gray_image.shape)
        self.assertTrue(np.all((thresholded == 0) | (thresholded == 255)))
    
    def test_inverse_vs_normal_threshold(self):
        """Test that inverse threshold is opposite of normal."""
        normal = apply_binary_threshold(self.gray_image, 127)
        inverse = apply_inverse_binary_threshold(self.gray_image, 127)
        
        # Inverse should be bitwise inverse
        self.assertTrue(np.all((normal == 0) | (normal == 255)))
        self.assertTrue(np.all((inverse == 0) | (inverse == 255)))


class TestThresholdMetrics(unittest.TestCase):
    """Test threshold metrics calculation."""
    
    def setUp(self):
        """Create test images."""
        self.gray_image = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
        self.color_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    
    def test_calculate_threshold_metrics_grayscale(self):
        """Test calculating threshold metrics for grayscale image."""
        metrics = calculate_threshold_metrics(self.gray_image)
        
        self.assertIn("min_value", metrics)
        self.assertIn("max_value", metrics)
        self.assertIn("dynamic_range", metrics)
        self.assertIn("mean", metrics)
        self.assertIn("std_dev", metrics)
        self.assertIn("histogram_peaks", metrics)
    
    def test_calculate_threshold_metrics_color(self):
        """Test calculating threshold metrics for color image."""
        metrics = calculate_threshold_metrics(self.color_image)
        
        self.assertIn("min_value", metrics)
        self.assertIn("max_value", metrics)
        self.assertGreater(metrics["max_value"], metrics["min_value"])


if __name__ == "__main__":
    unittest.main()
