"""Tests for advanced deblurring functionality for license plates."""

import unittest
import numpy as np
import cv2
import tempfile
from pathlib import Path

from app.processing.advanced_deblur import (
    apply_blind_deconvolution,
    apply_lucy_richardson_advanced,
    apply_total_variation_deblur,
    apply_super_resolution_upscale,
    apply_morphological_enhancement,
    apply_frequency_domain_deblur,
    apply_multi_scale_deblur,
    calculate_plate_visibility_score,
    recommend_deblur_method
)


class TestAdvancedDeblur(unittest.TestCase):
    """Test advanced deblurring methods."""
    
    def setUp(self):
        """Create test images."""
        # Create a simple blurred plate-like image
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
        
        # Create blurred version
        self.blurred_image = cv2.GaussianBlur(self.test_image, (9, 9), 2.0)
    
    def test_blind_deconvolution(self):
        """Test blind deconvolution."""
        result = apply_blind_deconvolution(
            self.blurred_image,
            kernel_size=(5, 5),
            iterations=20
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.blurred_image.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_lucy_richardson_advanced(self):
        """Test advanced Richardson-Lucy with different PSF types."""
        for psf_type in ["motion", "gaussian", "uniform"]:
            with self.subTest(psf_type=psf_type):
                result = apply_lucy_richardson_advanced(
                    self.blurred_image,
                    kernel_size=5,
                    iterations=10,
                    psf_type=psf_type
                )
                
                self.assertIsNotNone(result)
                self.assertEqual(result.shape, self.blurred_image.shape)
    
    def test_total_variation_deblur(self):
        """Test total variation deblurring."""
        result = apply_total_variation_deblur(
            self.blurred_image,
            strength=0.1,
            iterations=50
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.blurred_image.shape)
    
    def test_super_resolution_upscale(self):
        """Test super-resolution upscaling."""
        for scale in [2, 3, 4]:
            with self.subTest(scale=scale):
                result = apply_super_resolution_upscale(
                    self.blurred_image,
                    scale_factor=scale,
                    iterations=5
                )
                
                expected_h = self.blurred_image.shape[0] * scale
                expected_w = self.blurred_image.shape[1] * scale
                
                self.assertEqual(result.shape[0], expected_h)
                self.assertEqual(result.shape[1], expected_w)
    
    def test_morphological_enhancement(self):
        """Test morphological enhancement."""
        operations_list = [
            ["open"],
            ["close"],
            ["open", "close"],
            ["gradient"],
            ["tophat", "blackhat"]
        ]
        
        for ops in operations_list:
            with self.subTest(operations=ops):
                result = apply_morphological_enhancement(
                    self.blurred_image,
                    kernel_size=5,
                    operations=ops
                )
                
                self.assertIsNotNone(result)
    
    def test_frequency_domain_deblur(self):
        """Test frequency domain deblurring."""
        for blur_type in ["motion", "gaussian", "uniform"]:
            with self.subTest(blur_type=blur_type):
                result = apply_frequency_domain_deblur(
                    self.blurred_image,
                    blur_type=blur_type,
                    strength=1.0
                )
                
                self.assertIsNotNone(result)
                self.assertEqual(result.shape, self.blurred_image.shape)
    
    def test_multi_scale_deblur(self):
        """Test multi-scale deblurring."""
        result = apply_multi_scale_deblur(
            self.blurred_image,
            scales=[0.5, 1.0]
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, self.blurred_image.shape)
    
    def test_plate_visibility_score(self):
        """Test plate visibility scoring."""
        metrics = calculate_plate_visibility_score(self.test_image)
        
        self.assertIn("visibility_score", metrics)
        self.assertIn("sharpness", metrics)
        self.assertIn("contrast", metrics)
        self.assertIn("edge_density", metrics)
        self.assertIn("is_readable", metrics)
        
        # Original should be more readable than blurred
        self.assertGreater(metrics["visibility_score"], 0)
        self.assertLessEqual(metrics["visibility_score"], 100)
    
    def test_visibility_comparison(self):
        """Test that original is more visible than blurred."""
        original_metrics = calculate_plate_visibility_score(self.test_image)
        blurred_metrics = calculate_plate_visibility_score(self.blurred_image)
        
        # Original should have higher sharpness
        self.assertGreater(
            original_metrics["sharpness"],
            blurred_metrics["sharpness"]
        )
    
    def test_recommend_deblur_method(self):
        """Test deblur method recommendation."""
        method, scores = recommend_deblur_method(self.blurred_image)
        
        self.assertIn(method, ["motion", "gaussian", "uniform", "blind"])
        self.assertIsInstance(scores, dict)
        self.assertIn("motion", scores)
        self.assertIn("gaussian", scores)
        self.assertIn("uniform", scores)
        self.assertIn("blind", scores)
        
        # Scores should sum to 1 (normalized)
        total = sum(scores.values())
        self.assertAlmostEqual(total, 1.0, places=5)


class TestAdvancedDeblurEdgeCases(unittest.TestCase):
    """Test edge cases for advanced deblurring."""
    
    def test_invalid_image_none(self):
        """Test with None image."""
        with self.assertRaises(ValueError):
            apply_blind_deconvolution(None)
    
    def test_invalid_image_empty(self):
        """Test with empty array."""
        with self.assertRaises(ValueError):
            apply_blind_deconvolution(np.array([]))
    
    def test_super_resolution_invalid_scale(self):
        """Test super-resolution with invalid scale."""
        image = np.ones((50, 50), dtype=np.uint8) * 100
        
        with self.assertRaises(ValueError):
            apply_super_resolution_upscale(image, scale_factor=5)
    
    def test_grayscale_input(self):
        """Test all methods with grayscale input."""
        gray_image = np.ones((50, 50), dtype=np.uint8) * 100
        
        # Blind deconvolution
        result = apply_blind_deconvolution(gray_image, iterations=10)
        self.assertIsNotNone(result)
        
        # Advanced Richardson-Lucy
        result = apply_lucy_richardson_advanced(gray_image, iterations=5)
        self.assertIsNotNone(result)
        
        # Total variation
        result = apply_total_variation_deblur(gray_image, iterations=20)
        self.assertIsNotNone(result)


class TestAdvancedDeblurIntegration(unittest.TestCase):
    """Integration tests for advanced deblurring."""
    
    def setUp(self):
        """Create realistic blurred plate image."""
        # Create more realistic plate image
        self.plate = np.ones((80, 200, 3), dtype=np.uint8) * 220  # Yellow background
        
        # Add text
        cv2.putText(
            self.plate,
            "ABC-1234",
            (20, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 0),
            3
        )
        
        # Create motion blur
        size = 15
        kernel = np.zeros((size, size))
        kernel[size//2, :] = np.ones(size)
        kernel = kernel / size
        
        self.motion_blurred = cv2.filter2D(self.plate, -1, kernel)
        
        # Create Gaussian blur
        self.gaussian_blurred = cv2.GaussianBlur(self.plate, (21, 21), 3.0)
    
    def test_motion_blur_recovery_pipeline(self):
        """Test pipeline for motion blur recovery."""
        # Step 1: Frequency domain deblur for motion blur
        result = apply_frequency_domain_deblur(
            self.motion_blurred,
            blur_type="motion",
            strength=1.5
        )
        
        # Step 2: Super-resolution
        result = apply_super_resolution_upscale(result, scale_factor=2, iterations=5)
        
        # Step 3: Morphological enhancement
        result = apply_morphological_enhancement(
            result,
            kernel_size=5,
            operations=["close", "open"]
        )
        
        self.assertIsNotNone(result)
        self.assertGreater(result.shape[0], self.motion_blurred.shape[0])
    
    def test_gaussian_blur_recovery_pipeline(self):
        """Test pipeline for Gaussian blur recovery."""
        # Step 1: Advanced Richardson-Lucy
        result = apply_lucy_richardson_advanced(
            self.gaussian_blurred,
            kernel_size=7,
            iterations=15,
            psf_type="gaussian"
        )
        
        # Step 2: Total variation deblur
        result = apply_total_variation_deblur(result, strength=0.15, iterations=50)
        
        # Step 3: Morphological enhancement
        result = apply_morphological_enhancement(
            result,
            kernel_size=3,
            operations=["open"]
        )
        
        self.assertIsNotNone(result)
    
    def test_automatic_deblur_selection(self):
        """Test automatic deblur method selection."""
        # Recommend method for motion blur
        method_motion, _ = recommend_deblur_method(self.motion_blurred)
        self.assertIn(method_motion, ["motion", "gaussian", "uniform", "blind"])
        
        # Recommend method for Gaussian blur
        method_gaussian, _ = recommend_deblur_method(self.gaussian_blurred)
        self.assertIn(method_gaussian, ["motion", "gaussian", "uniform", "blind"])
    
    def test_visibility_improvement(self):
        """Test that deblurring improves visibility."""
        # Original visibility
        original_score = calculate_plate_visibility_score(self.plate)
        
        # Blurred visibility
        blurred_score = calculate_plate_visibility_score(self.gaussian_blurred)
        
        # Deblurred visibility
        deblurred = apply_lucy_richardson_advanced(
            self.gaussian_blurred,
            iterations=10,
            psf_type="gaussian"
        )
        deblurred_score = calculate_plate_visibility_score(deblurred)
        
        # Verify improvement
        self.assertGreater(original_score["visibility_score"], blurred_score["visibility_score"])
        self.assertGreater(deblurred_score["visibility_score"], blurred_score["visibility_score"])


if __name__ == "__main__":
    unittest.main()
