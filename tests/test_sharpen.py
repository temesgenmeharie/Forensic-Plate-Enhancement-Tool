"""Tests for sharpening module."""

import unittest

import numpy as np

from app.processing.sharpen import (
    apply_unsharp_mask, apply_laplacian_sharpening,
    apply_high_pass_filter, apply_kernel_sharpening,
    calculate_sharpness_metrics
)


class TestUnsharpMask(unittest.TestCase):
    """Test unsharp masking."""
    
    def setUp(self):
        """Create test image."""
        self.gray_image = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
        self.color_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    
    def test_unsharp_mask_grayscale(self):
        """Test unsharp mask on grayscale image."""
        sharpened = apply_unsharp_mask(self.gray_image)
        
        self.assertEqual(sharpened.shape, self.gray_image.shape)
        self.assertEqual(sharpened.dtype, self.gray_image.dtype)
    
    def test_unsharp_mask_color(self):
        """Test unsharp mask on color image."""
        sharpened = apply_unsharp_mask(self.color_image)
        
        self.assertEqual(sharpened.shape, self.color_image.shape)
    
    def test_unsharp_mask_custom_params(self):
        """Test unsharp mask with custom parameters."""
        sharpened = apply_unsharp_mask(
            self.gray_image,
            sigma=2.0,
            amount=2.0,
            threshold=10.0
        )
        
        self.assertEqual(sharpened.shape, self.gray_image.shape)
    
    def test_unsharp_mask_invalid_params(self):
        """Test unsharp mask with invalid parameters."""
        with self.assertRaises(ValueError):
            apply_unsharp_mask(self.gray_image, sigma=0)
        
        with self.assertRaises(ValueError):
            apply_unsharp_mask(self.gray_image, amount=-1)
        
        with self.assertRaises(ValueError):
            apply_unsharp_mask(self.gray_image, threshold=300)


class TestLaplacianSharpening(unittest.TestCase):
    """Test Laplacian sharpening."""
    
    def setUp(self):
        """Create test image."""
        self.gray_image = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
    
    def test_laplacian_sharpening(self):
        """Test Laplacian sharpening."""
        sharpened = apply_laplacian_sharpening(self.gray_image)
        
        self.assertEqual(sharpened.shape, self.gray_image.shape)


class TestHighPassFilter(unittest.TestCase):
    """Test high-pass filter."""
    
    def setUp(self):
        """Create test image."""
        self.gray_image = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
    
    def test_high_pass_filter(self):
        """Test high-pass filter."""
        enhanced = apply_high_pass_filter(self.gray_image)
        
        self.assertEqual(enhanced.shape, self.gray_image.shape)
    
    def test_high_pass_filter_custom_params(self):
        """Test high-pass filter with custom parameters."""
        enhanced = apply_high_pass_filter(
            self.gray_image,
            sigma=3.0,
            amount=1.5
        )
        
        self.assertEqual(enhanced.shape, self.gray_image.shape)


class TestKernelSharpening(unittest.TestCase):
    """Test kernel-based sharpening."""
    
    def setUp(self):
        """Create test image."""
        self.gray_image = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
    
    def test_kernel_sharpening_sharp(self):
        """Test kernel sharpening with sharp kernel."""
        sharpened = apply_kernel_sharpening(self.gray_image, kernel_type="sharp")
        
        self.assertEqual(sharpened.shape, self.gray_image.shape)
    
    def test_kernel_sharpening_stronger(self):
        """Test kernel sharpening with stronger kernel."""
        sharpened = apply_kernel_sharpening(self.gray_image, kernel_type="stronger")
        
        self.assertEqual(sharpened.shape, self.gray_image.shape)
    
    def test_kernel_sharpening_edge_detect(self):
        """Test kernel sharpening with edge detect kernel."""
        sharpened = apply_kernel_sharpening(self.gray_image, kernel_type="edge_detect")
        
        self.assertEqual(sharpened.shape, self.gray_image.shape)
    
    def test_kernel_sharpening_invalid_kernel(self):
        """Test kernel sharpening with invalid kernel type."""
        with self.assertRaises(ValueError):
            apply_kernel_sharpening(self.gray_image, kernel_type="invalid")


class TestSharpnessMetrics(unittest.TestCase):
    """Test sharpness metrics calculation."""
    
    def setUp(self):
        """Create test images."""
        # Create a sharp image
        self.sharp_image = np.zeros((100, 100), dtype=np.uint8)
        self.sharp_image[40:60, 40:60] = 255
        
        # Create a blurry image
        from app.processing.denoise import apply_gaussian_blur
        self.blurry_image = apply_gaussian_blur(self.sharp_image, kernel_size=11)
    
    def test_calculate_sharpness_metrics(self):
        """Test calculating sharpness metrics."""
        metrics = calculate_sharpness_metrics(self.sharp_image)
        
        self.assertIn("laplacian_variance", metrics)
        self.assertIn("sobel_variance", metrics)
        self.assertIn("gradient_mean", metrics)
        self.assertIn("is_sharp", metrics)
    
    def test_sharpness_metrics_detect_sharpness(self):
        """Test that sharpness metrics detect differences."""
        metrics_sharp = calculate_sharpness_metrics(self.sharp_image)
        metrics_blurry = calculate_sharpness_metrics(self.blurry_image)
        
        # Sharp image should have higher Laplacian variance
        self.assertGreater(
            metrics_sharp["laplacian_variance"],
            metrics_blurry["laplacian_variance"]
        )


if __name__ == "__main__":
    unittest.main()
