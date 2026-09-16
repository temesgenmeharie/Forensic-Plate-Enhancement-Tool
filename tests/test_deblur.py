"""Tests for deblurring module."""

import unittest

import cv2
import numpy as np

from app.processing.deblur import (
    apply_wiener_filter, apply_richardson_lucy_deconvolution,
    apply_motion_blur_removal, apply_laplacian_sharpening_deblur,
    calculate_blur_metrics
)


class TestWienerFilter(unittest.TestCase):
    """Test Wiener filtering."""
    
    def setUp(self):
        """Create test images."""
        self.gray_image = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
        self.color_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        
        # Create a blurry image
        self.blurry_image = cv2.GaussianBlur(self.gray_image, (11, 11), 2.0)
    
    def test_wiener_filter_grayscale(self):
        """Test Wiener filter on grayscale image."""
        deblurred = apply_wiener_filter(self.blurry_image)
        
        self.assertEqual(deblurred.shape, self.blurry_image.shape)
        self.assertTrue(np.all(deblurred >= 0))
        self.assertTrue(np.all(deblurred <= 255))
    
    def test_wiener_filter_color(self):
        """Test Wiener filter on color image."""
        deblurred = apply_wiener_filter(self.color_image)
        
        # Should return grayscale
        self.assertEqual(len(deblurred.shape), 2)
    
    def test_wiener_filter_custom_params(self):
        """Test Wiener filter with custom parameters."""
        deblurred = apply_wiener_filter(
            self.blurry_image,
            noise_variance=20.0,
            kernel_size=7
        )
        
        self.assertEqual(deblurred.shape, self.blurry_image.shape)
    
    def test_wiener_filter_invalid_noise_variance(self):
        """Test Wiener filter with invalid noise variance."""
        with self.assertRaises(ValueError):
            apply_wiener_filter(self.blurry_image, noise_variance=-1)
    
    def test_wiener_filter_invalid_kernel_size(self):
        """Test Wiener filter with invalid kernel size."""
        with self.assertRaises(ValueError):
            apply_wiener_filter(self.blurry_image, kernel_size=0)
        
        with self.assertRaises(ValueError):
            apply_wiener_filter(self.blurry_image, kernel_size=6)  # Even


class TestRichardsonLucy(unittest.TestCase):
    """Test Richardson-Lucy deconvolution."""
    
    def setUp(self):
        """Create test images."""
        self.gray_image = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
        
        # Create a blurry image
        self.blurry_image = cv2.GaussianBlur(self.gray_image, (9, 9), 1.5)
    
    def test_richardson_lucy_basic(self):
        """Test basic Richardson-Lucy deconvolution."""
        deblurred = apply_richardson_lucy_deconvolution(self.blurry_image)
        
        self.assertEqual(deblurred.shape, self.blurry_image.shape)
        self.assertTrue(np.all(deblurred >= 0))
        self.assertTrue(np.all(deblurred <= 255))
    
    def test_richardson_lucy_custom_params(self):
        """Test Richardson-Lucy with custom parameters."""
        deblurred = apply_richardson_lucy_deconvolution(
            self.blurry_image,
            kernel_size=7,
            iterations=20,
            regularization=0.5
        )
        
        self.assertEqual(deblurred.shape, self.blurry_image.shape)
    
    def test_richardson_lucy_invalid_kernel_size(self):
        """Test Richardson-Lucy with invalid kernel size."""
        with self.assertRaises(ValueError):
            apply_richardson_lucy_deconvolution(self.blurry_image, kernel_size=0)
        
        with self.assertRaises(ValueError):
            apply_richardson_lucy_deconvolution(self.blurry_image, kernel_size=4)
    
    def test_richardson_lucy_invalid_iterations(self):
        """Test Richardson-Lucy with invalid iterations."""
        with self.assertRaises(ValueError):
            apply_richardson_lucy_deconvolution(self.blurry_image, iterations=0)
    
    def test_richardson_lucy_invalid_regularization(self):
        """Test Richardson-Lucy with invalid regularization."""
        with self.assertRaises(ValueError):
            apply_richardson_lucy_deconvolution(self.blurry_image, regularization=-0.1)
        
        with self.assertRaises(ValueError):
            apply_richardson_lucy_deconvolution(self.blurry_image, regularization=1.5)


class TestMotionBlurRemoval(unittest.TestCase):
    """Test motion blur removal."""
    
    def setUp(self):
        """Create test images."""
        self.gray_image = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
        
        # Create a motion-blurred image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self.motion_blurred = cv2.filter2D(self.gray_image, -1, kernel)
    
    def test_motion_blur_removal_basic(self):
        """Test basic motion blur removal."""
        deblurred = apply_motion_blur_removal(self.motion_blurred)
        
        self.assertEqual(deblurred.shape, self.motion_blurred.shape)
        self.assertTrue(np.all(deblurred >= 0))
        self.assertTrue(np.all(deblurred <= 255))
    
    def test_motion_blur_removal_custom_angle(self):
        """Test motion blur removal with custom angle."""
        deblurred = apply_motion_blur_removal(
            self.motion_blurred,
            kernel_size=13,
            angle=45.0,
            iterations=3
        )
        
        self.assertEqual(deblurred.shape, self.motion_blurred.shape)
    
    def test_motion_blur_removal_horizontal(self):
        """Test motion blur removal for horizontal motion."""
        deblurred = apply_motion_blur_removal(
            self.motion_blurred,
            angle=0.0
        )
        
        self.assertEqual(deblurred.shape, self.motion_blurred.shape)
    
    def test_motion_blur_removal_vertical(self):
        """Test motion blur removal for vertical motion."""
        deblurred = apply_motion_blur_removal(
            self.motion_blurred,
            angle=90.0
        )
        
        self.assertEqual(deblurred.shape, self.motion_blurred.shape)
    
    def test_motion_blur_removal_invalid_kernel_size(self):
        """Test motion blur removal with invalid kernel size."""
        with self.assertRaises(ValueError):
            apply_motion_blur_removal(self.motion_blurred, kernel_size=0)
        
        with self.assertRaises(ValueError):
            apply_motion_blur_removal(self.motion_blurred, kernel_size=14)
    
    def test_motion_blur_removal_invalid_angle(self):
        """Test motion blur removal with invalid angle."""
        with self.assertRaises(ValueError):
            apply_motion_blur_removal(self.motion_blurred, angle=-1)
        
        with self.assertRaises(ValueError):
            apply_motion_blur_removal(self.motion_blurred, angle=181)


class TestLaplacianSharpening(unittest.TestCase):
    """Test Laplacian-based deblurring."""
    
    def setUp(self):
        """Create test image."""
        self.gray_image = np.random.randint(50, 200, (480, 640), dtype=np.uint8)
        self.blurry_image = cv2.GaussianBlur(self.gray_image, (11, 11), 2.0)
    
    def test_laplacian_sharpening_deblur(self):
        """Test Laplacian sharpening for deblurring."""
        sharpened = apply_laplacian_sharpening_deblur(self.blurry_image)
        
        self.assertEqual(sharpened.shape, self.blurry_image.shape)
        self.assertTrue(np.all(sharpened >= 0))
        self.assertTrue(np.all(sharpened <= 255))
    
    def test_laplacian_sharpening_custom_strength(self):
        """Test Laplacian sharpening with custom strength."""
        sharpened = apply_laplacian_sharpening_deblur(
            self.blurry_image,
            strength=2.0
        )
        
        self.assertEqual(sharpened.shape, self.blurry_image.shape)


class TestBlurMetrics(unittest.TestCase):
    """Test blur metric calculation."""
    
    def setUp(self):
        """Create test images."""
        # Create a sharp image
        self.sharp_image = np.zeros((100, 100), dtype=np.uint8)
        self.sharp_image[40:60, 40:60] = 255
        
        # Create a blurry image
        self.blurry_image = cv2.GaussianBlur(self.sharp_image, (15, 15), 3.0)
    
    def test_calculate_blur_metrics_sharp(self):
        """Test calculating blur metrics for sharp image."""
        metrics = calculate_blur_metrics(self.sharp_image)
        
        self.assertIn("laplacian_variance", metrics)
        self.assertIn("edge_magnitude", metrics)
        self.assertIn("high_frequency_content", metrics)
        self.assertIn("is_blurry", metrics)
        self.assertIn("blur_severity", metrics)
    
    def test_calculate_blur_metrics_blurry(self):
        """Test calculating blur metrics for blurry image."""
        metrics = calculate_blur_metrics(self.blurry_image)
        
        self.assertIn("laplacian_variance", metrics)
        self.assertIn("blur_severity", metrics)
    
    def test_blur_metrics_detect_blur_difference(self):
        """Test that blur metrics detect differences between sharp and blurry."""
        metrics_sharp = calculate_blur_metrics(self.sharp_image)
        metrics_blurry = calculate_blur_metrics(self.blurry_image)
        
        # Sharp image should have higher Laplacian variance
        self.assertGreater(
            metrics_sharp["laplacian_variance"],
            metrics_blurry["laplacian_variance"]
        )
        
        # Sharp image should have lower blur severity
        self.assertLess(
            metrics_sharp["blur_severity"],
            metrics_blurry["blur_severity"]
        )
    
    def test_blur_metrics_range_validation(self):
        """Test that blur severity is in valid range."""
        metrics = calculate_blur_metrics(self.blurry_image)
        
        self.assertGreaterEqual(metrics["blur_severity"], 0)
        self.assertLessEqual(metrics["blur_severity"], 100)


if __name__ == "__main__":
    unittest.main()
