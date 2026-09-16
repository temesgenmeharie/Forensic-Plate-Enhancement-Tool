"""Tests for denoising module."""

import unittest

import numpy as np

from app.processing.denoise import (
    apply_bilateral_filter, apply_non_local_means_denoise,
    apply_morphological_opening, apply_gaussian_blur,
    calculate_noise_metrics
)


class TestBilateralFilter(unittest.TestCase):
    """Test bilateral filtering."""
    
    def setUp(self):
        """Create test images."""
        self.gray_image = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        self.color_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    def test_bilateral_filter_grayscale(self):
        """Test bilateral filter on grayscale image."""
        filtered = apply_bilateral_filter(self.gray_image)
        
        self.assertEqual(filtered.shape, self.gray_image.shape)
        self.assertTrue(np.all(filtered >= 0))
        self.assertTrue(np.all(filtered <= 255))
    
    def test_bilateral_filter_color(self):
        """Test bilateral filter on color image."""
        filtered = apply_bilateral_filter(self.color_image)
        
        self.assertEqual(filtered.shape, self.color_image.shape)
    
    def test_bilateral_filter_custom_params(self):
        """Test bilateral filter with custom parameters."""
        filtered = apply_bilateral_filter(
            self.gray_image,
            diameter=15,
            sigma_color=100,
            sigma_space=100
        )
        
        self.assertEqual(filtered.shape, self.gray_image.shape)
    
    def test_bilateral_filter_invalid_diameter(self):
        """Test bilateral filter with invalid diameter."""
        with self.assertRaises(ValueError):
            apply_bilateral_filter(self.gray_image, diameter=0)
        
        with self.assertRaises(ValueError):
            apply_bilateral_filter(self.gray_image, diameter=10)  # Even number


class TestNonLocalMeans(unittest.TestCase):
    """Test non-local means denoising."""
    
    def setUp(self):
        """Create test images."""
        self.gray_image = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        self.color_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    def test_nlm_denoise_grayscale(self):
        """Test NLM denoising on grayscale image."""
        denoised = apply_non_local_means_denoise(self.gray_image, h=10)
        
        self.assertEqual(denoised.shape, self.gray_image.shape)
    
    def test_nlm_denoise_color(self):
        """Test NLM denoising on color image."""
        denoised = apply_non_local_means_denoise(self.color_image, h=10)
        
        self.assertEqual(denoised.shape, self.color_image.shape)
    
    def test_nlm_denoise_invalid_h(self):
        """Test NLM with invalid h parameter."""
        with self.assertRaises(ValueError):
            apply_non_local_means_denoise(self.gray_image, h=0)


class TestMorphologicalOpening(unittest.TestCase):
    """Test morphological opening."""
    
    def setUp(self):
        """Create test image."""
        self.gray_image = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
    
    def test_morphological_opening(self):
        """Test morphological opening."""
        opened = apply_morphological_opening(self.gray_image)
        
        self.assertEqual(opened.shape, self.gray_image.shape)
    
    def test_morphological_opening_custom_kernel(self):
        """Test morphological opening with custom kernel."""
        opened = apply_morphological_opening(self.gray_image, kernel_size=7)
        
        self.assertEqual(opened.shape, self.gray_image.shape)


class TestGaussianBlur(unittest.TestCase):
    """Test Gaussian blur."""
    
    def setUp(self):
        """Create test image."""
        self.gray_image = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
    
    def test_gaussian_blur(self):
        """Test Gaussian blur."""
        blurred = apply_gaussian_blur(self.gray_image)
        
        self.assertEqual(blurred.shape, self.gray_image.shape)
    
    def test_gaussian_blur_custom_params(self):
        """Test Gaussian blur with custom parameters."""
        blurred = apply_gaussian_blur(
            self.gray_image,
            kernel_size=7,
            sigma=2.0
        )
        
        self.assertEqual(blurred.shape, self.gray_image.shape)


class TestNoiseMetrics(unittest.TestCase):
    """Test noise metrics calculation."""
    
    def setUp(self):
        """Create test images."""
        # Create a sharp image
        self.sharp_image = np.zeros((100, 100), dtype=np.uint8)
        self.sharp_image[40:60, 40:60] = 255
        
        # Create a blurry image
        self.blurry_image = apply_gaussian_blur(self.sharp_image, kernel_size=11)
    
    def test_calculate_noise_metrics(self):
        """Test calculating noise metrics."""
        metrics = calculate_noise_metrics(self.sharp_image)
        
        self.assertIn("laplacian_variance", metrics)
        self.assertIn("std_deviation", metrics)
        self.assertIn("is_sharp", metrics)
    
    def test_noise_metrics_sharpness_detection(self):
        """Test that noise metrics detect sharpness differences."""
        metrics_sharp = calculate_noise_metrics(self.sharp_image)
        metrics_blurry = calculate_noise_metrics(self.blurry_image)
        
        # Sharp image should have higher Laplacian variance
        self.assertGreater(
            metrics_sharp["laplacian_variance"],
            metrics_blurry["laplacian_variance"]
        )


if __name__ == "__main__":
    unittest.main()
