"""Tests for image resizing and upscaling."""

import unittest

import numpy as np

from app.processing.resize import (
    InterpolationMethod, get_interpolation_method, upscale_image,
    upscale_image_multiple, resize_to_size, compare_interpolation_methods,
    calculate_upscaling_statistics
)


class TestInterpolationMethod(unittest.TestCase):
    """Test interpolation method selection."""
    
    def test_get_interpolation_method(self):
        """Test getting interpolation method by name."""
        method = get_interpolation_method("LANCZOS4")
        self.assertIsNotNone(method)
    
    def test_get_interpolation_method_case_insensitive(self):
        """Test that method lookup is case-insensitive."""
        method1 = get_interpolation_method("LANCZOS4")
        method2 = get_interpolation_method("lanczos4")
        self.assertEqual(method1, method2)
    
    def test_get_interpolation_method_invalid(self):
        """Test that invalid method raises ValueError."""
        with self.assertRaises(ValueError):
            get_interpolation_method("INVALID_METHOD")


class TestUpscaling(unittest.TestCase):
    """Test image upscaling functionality."""
    
    def setUp(self):
        """Create test image."""
        self.test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    def test_upscale_2x(self):
        """Test 2x upscaling."""
        upscaled = upscale_image(self.test_image, 2.0)
        
        self.assertEqual(upscaled.shape[0], 960)
        self.assertEqual(upscaled.shape[1], 1280)
    
    def test_upscale_4x(self):
        """Test 4x upscaling."""
        upscaled = upscale_image(self.test_image, 4.0)
        
        self.assertEqual(upscaled.shape[0], 1920)
        self.assertEqual(upscaled.shape[1], 2560)
    
    def test_upscale_8x(self):
        """Test 8x upscaling."""
        upscaled = upscale_image(self.test_image, 8.0)
        
        self.assertEqual(upscaled.shape[0], 3840)
        self.assertEqual(upscaled.shape[1], 5120)
    
    def test_upscale_identity(self):
        """Test that 1.0 scale factor returns copy."""
        upscaled = upscale_image(self.test_image, 1.0)
        
        self.assertEqual(upscaled.shape, self.test_image.shape)
        np.testing.assert_array_equal(upscaled, self.test_image)
    
    def test_upscale_invalid_scale(self):
        """Test that invalid scale factor raises ValueError."""
        with self.assertRaises(ValueError):
            upscale_image(self.test_image, 0)
        
        with self.assertRaises(ValueError):
            upscale_image(self.test_image, -1)
    
    def test_upscale_invalid_image(self):
        """Test that invalid image raises ValueError."""
        with self.assertRaises(ValueError):
            upscale_image(None, 2.0)
        
        with self.assertRaises(ValueError):
            upscale_image(np.array([]), 2.0)
    
    def test_upscale_with_different_interpolations(self):
        """Test upscaling with different interpolation methods."""
        for method in ["NEAREST", "LINEAR", "CUBIC", "LANCZOS4"]:
            upscaled = upscale_image(self.test_image, 2.0, method)
            self.assertEqual(upscaled.shape[0], 960)
            self.assertEqual(upscaled.shape[1], 1280)


class TestUpscaleMultiple(unittest.TestCase):
    """Test multiple upscaling generation."""
    
    def setUp(self):
        """Create test image."""
        self.test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    def test_upscale_multiple_default(self):
        """Test generating multiple upscales with default scales."""
        results = upscale_image_multiple(self.test_image)
        
        self.assertEqual(len(results), 3)
        self.assertIn(2.0, results)
        self.assertIn(4.0, results)
        self.assertIn(8.0, results)
    
    def test_upscale_multiple_custom_scales(self):
        """Test generating multiple upscales with custom scales."""
        scales = [1.5, 3.0, 5.0]
        results = upscale_image_multiple(self.test_image, scale_factors=scales)
        
        self.assertEqual(len(results), 3)
        for scale in scales:
            self.assertIn(scale, results)


class TestResize(unittest.TestCase):
    """Test image resizing functionality."""
    
    def setUp(self):
        """Create test image."""
        self.test_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    def test_resize_to_size(self):
        """Test resizing to specific size."""
        resized = resize_to_size(self.test_image, 800, 600)
        
        self.assertEqual(resized.shape[0], 600)
        self.assertEqual(resized.shape[1], 800)
    
    def test_resize_to_size_keep_aspect_ratio(self):
        """Test resizing with aspect ratio preservation."""
        resized = resize_to_size(self.test_image, 800, 600, keep_aspect_ratio=True)
        
        # Should fit within target size
        self.assertLessEqual(resized.shape[1], 800)
        self.assertLessEqual(resized.shape[0], 600)
    
    def test_resize_invalid_dimensions(self):
        """Test that invalid dimensions raise ValueError."""
        with self.assertRaises(ValueError):
            resize_to_size(self.test_image, 0, 100)
        
        with self.assertRaises(ValueError):
            resize_to_size(self.test_image, 100, -1)


class TestCompareInterpolation(unittest.TestCase):
    """Test interpolation method comparison."""
    
    def setUp(self):
        """Create test image."""
        self.test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    def test_compare_interpolation_methods(self):
        """Test comparing interpolation methods."""
        results = compare_interpolation_methods(self.test_image, scale_factor=2.0)
        
        # Should have results for each method
        self.assertGreater(len(results), 0)
        
        # All results should have same shape
        result_shapes = [img.shape for img in results.values()]
        self.assertTrue(all(shape == result_shapes[0] for shape in result_shapes))


class TestUpscalingStatistics(unittest.TestCase):
    """Test upscaling statistics calculation."""
    
    def setUp(self):
        """Create test images."""
        self.original = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        self.upscaled = upscale_image(self.original, 2.0)
    
    def test_calculate_statistics(self):
        """Test calculating upscaling statistics."""
        stats = calculate_upscaling_statistics(self.original, self.upscaled)
        
        self.assertIn("original_dimensions", stats)
        self.assertIn("upscaled_dimensions", stats)
        self.assertIn("scale_factor_width", stats)
        self.assertIn("pixel_increase_ratio", stats)
        
        self.assertEqual(stats["scale_factor_width"], 2.0)
        self.assertEqual(stats["pixel_increase_ratio"], 4.0)


if __name__ == "__main__":
    unittest.main()
