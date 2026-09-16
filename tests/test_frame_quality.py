"""Tests for frame quality assessment module."""

import unittest

import cv2
import numpy as np

from app.video.frame_quality import FrameQualityMetrics, FrameSelector


class TestFrameQualityMetrics(unittest.TestCase):
    """Test frame quality metric calculation."""
    
    def setUp(self):
        """Create test frames."""
        # Create a sharp frame
        self.sharp_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.sharp_frame[200:280, 300:340] = [0, 255, 0]  # Green rectangle
        
        # Create a blurry frame
        self.blurry_frame = cv2.GaussianBlur(self.sharp_frame, (21, 21), 3.0)
        
        # Create a bright frame
        self.bright_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # Create a dark frame
        self.dark_frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
    
    def test_calculate_sharpness(self):
        """Test sharpness calculation."""
        sharpness = FrameQualityMetrics.calculate_sharpness(self.sharp_frame)
        
        self.assertIsInstance(sharpness, float)
        self.assertGreater(sharpness, 0)
    
    def test_sharpness_detection(self):
        """Test that sharp frames have higher sharpness score."""
        sharpness_sharp = FrameQualityMetrics.calculate_sharpness(self.sharp_frame)
        sharpness_blurry = FrameQualityMetrics.calculate_sharpness(self.blurry_frame)
        
        self.assertGreater(sharpness_sharp, sharpness_blurry)
    
    def test_calculate_brightness(self):
        """Test brightness calculation."""
        brightness = FrameQualityMetrics.calculate_brightness(self.bright_frame)
        
        self.assertGreater(brightness, 100)
    
    def test_brightness_comparison(self):
        """Test brightness comparison between frames."""
        brightness_bright = FrameQualityMetrics.calculate_brightness(self.bright_frame)
        brightness_dark = FrameQualityMetrics.calculate_brightness(self.dark_frame)
        
        self.assertGreater(brightness_bright, brightness_dark)
    
    def test_calculate_contrast(self):
        """Test contrast calculation."""
        contrast = FrameQualityMetrics.calculate_contrast(self.sharp_frame)
        
        self.assertGreater(contrast, 0)
    
    def test_calculate_edge_density(self):
        """Test edge density calculation."""
        edge_density = FrameQualityMetrics.calculate_edge_density(self.sharp_frame)
        
        self.assertGreaterEqual(edge_density, 0)
        self.assertLessEqual(edge_density, 1)
    
    def test_calculate_blur_score(self):
        """Test blur score calculation."""
        blur_score = FrameQualityMetrics.calculate_blur_score(self.sharp_frame)
        
        self.assertGreaterEqual(blur_score, 0)
        self.assertLessEqual(blur_score, 100)
    
    def test_blur_score_range(self):
        """Test that blur score is in valid range."""
        blur_score_sharp = FrameQualityMetrics.calculate_blur_score(self.sharp_frame)
        blur_score_blurry = FrameQualityMetrics.calculate_blur_score(self.blurry_frame)
        
        self.assertLess(blur_score_sharp, blur_score_blurry)
    
    def test_calculate_saturation(self):
        """Test saturation calculation."""
        saturation = FrameQualityMetrics.calculate_saturation(self.sharp_frame)
        
        self.assertGreaterEqual(saturation, 0)
        self.assertLessEqual(saturation, 255)
    
    def test_calculate_all_metrics(self):
        """Test calculating all metrics at once."""
        metrics = FrameQualityMetrics.calculate_all_metrics(self.sharp_frame)
        
        self.assertIn("sharpness", metrics)
        self.assertIn("brightness", metrics)
        self.assertIn("contrast", metrics)
        self.assertIn("edge_density", metrics)
        self.assertIn("blur_score", metrics)
        self.assertIn("saturation", metrics)
    
    def test_metrics_on_grayscale(self):
        """Test metrics on grayscale frame."""
        gray_frame = cv2.cvtColor(self.sharp_frame, cv2.COLOR_BGR2GRAY)
        metrics = FrameQualityMetrics.calculate_all_metrics(gray_frame)
        
        self.assertIn("sharpness", metrics)
        self.assertEqual(metrics["saturation"], 0)  # Grayscale has no saturation


class TestFrameSelector(unittest.TestCase):
    """Test frame selection functionality."""
    
    def setUp(self):
        """Create test frames with metrics."""
        # Create multiple test frames with varying contrast
        frames = []
        
        for i in range(5):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add checkerboard pattern for contrast
            for y in range(0, 480, 20):
                for x in range(0, 640, 20):
                    if (y // 20 + x // 20) % 2 == 0:
                        frame[y:y+20, x:x+20] = [50 + i * 30, 50 + i * 30, 50 + i * 30]
                    else:
                        frame[y:y+20, x:x+20] = [150 + i * 20, 150 + i * 20, 150 + i * 20]
            
            metrics = FrameQualityMetrics.calculate_all_metrics(frame)
            frames.append((i, frame, metrics))
        
        self.frames_with_metrics = frames
        self.selector = FrameSelector(frames)
    
    def test_select_by_sharpness(self):
        """Test selecting frames by sharpness."""
        selected = self.selector.select_by_sharpness(top_n=2)
        
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(selected[0]), 2)  # (frame_num, frame)
    
    def test_select_by_brightness_range(self):
        """Test selecting frames by brightness range."""
        selected = self.selector.select_by_brightness_range(
            min_brightness=50,
            max_brightness=150
        )
        
        self.assertGreater(len(selected), 0)
    
    def test_select_by_contrast(self):
        """Test selecting frames by contrast."""
        selected = self.selector.select_by_contrast(threshold=1)  # Lower threshold
        
        self.assertGreater(len(selected), 0)
    
    def test_select_best_composite(self):
        """Test selecting frames by composite quality."""
        selected = self.selector.select_best_composite(top_n=3)
        
        self.assertEqual(len(selected), 3)


if __name__ == "__main__":
    unittest.main()
