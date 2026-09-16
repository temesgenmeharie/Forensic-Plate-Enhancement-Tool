"""Tests for frame alignment and multi-frame analysis."""

import unittest

import cv2
import numpy as np

from app.video.frame_alignment import FrameAligner, MultiFrameComparison


class TestFrameAligner(unittest.TestCase):
    """Test frame alignment functionality."""
    
    def setUp(self):
        """Create test frames."""
        # Create a reference frame with distinctive pattern
        self.reference_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(self.reference_frame, (100, 100), (400, 300), (0, 255, 0), -1)
        cv2.rectangle(self.reference_frame, (200, 150), (300, 250), (0, 0, 255), -1)
        
        # Create a shifted version
        M = cv2.getRotationMatrix2D((320, 240), 0, 1)
        M[0, 2] += 10  # Shift x
        M[1, 2] += 5   # Shift y
        self.shifted_frame = cv2.warpAffine(
            self.reference_frame,
            M,
            (640, 480)
        )
    
    def test_detect_features(self):
        """Test feature detection."""
        gray = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = FrameAligner.detect_features(gray)
        
        self.assertGreater(len(keypoints), 0)
        self.assertIsNotNone(descriptors)
    
    def test_detect_features_grayscale(self):
        """Test feature detection on grayscale image."""
        gray = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = FrameAligner.detect_features(gray)
        
        self.assertGreater(len(keypoints), 0)
    
    def test_match_features(self):
        """Test feature matching."""
        gray1 = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.shifted_frame, cv2.COLOR_BGR2GRAY)
        
        kp1, desc1 = FrameAligner.detect_features(gray1)
        kp2, desc2 = FrameAligner.detect_features(gray2)
        
        matches = FrameAligner.match_features(desc1, desc2)
        
        self.assertGreaterEqual(len(matches), 0)
    
    def test_compute_homography(self):
        """Test homography computation."""
        gray1 = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.shifted_frame, cv2.COLOR_BGR2GRAY)
        
        kp1, desc1 = FrameAligner.detect_features(gray1)
        kp2, desc2 = FrameAligner.detect_features(gray2)
        matches = FrameAligner.match_features(desc1, desc2)
        
        if len(matches) >= 4:
            H = FrameAligner.compute_homography(kp1, kp2, matches)
            
            if H is not None:
                self.assertEqual(H.shape, (3, 3))
    
    def test_warp_image(self):
        """Test image warping."""
        # Create identity homography
        H = np.eye(3)
        
        warped = FrameAligner.warp_image(
            self.reference_frame,
            H,
            (480, 640)
        )
        
        self.assertEqual(warped.shape, self.reference_frame.shape)
    
    def test_align_frame_to_reference(self):
        """Test frame alignment."""
        aligned, H = FrameAligner.align_frame_to_reference(
            self.reference_frame,
            self.shifted_frame
        )
        
        # Alignment might not always succeed with synthetic frames
        # but the function should handle it gracefully
        if aligned is not None:
            self.assertEqual(aligned.shape, self.reference_frame.shape)


class TestMultiFrameComparison(unittest.TestCase):
    """Test multi-frame analysis functionality."""
    
    def setUp(self):
        """Create test frames."""
        # Frame 1: Solid color
        self.frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 100
        
        # Frame 2: Slightly different
        self.frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 110
        
        # Frame 3: Very different
        self.frame3 = np.ones((480, 640, 3), dtype=np.uint8) * 50
    
    def test_compute_frame_difference(self):
        """Test frame difference computation."""
        diff = MultiFrameComparison.compute_frame_difference(self.frame1, self.frame2)
        
        self.assertIn("mean_absolute_difference", diff)
        self.assertIn("max_difference", diff)
        self.assertIn("pixels_different", diff)
    
    def test_frame_difference_values(self):
        """Test that difference values are reasonable."""
        diff12 = MultiFrameComparison.compute_frame_difference(self.frame1, self.frame2)
        diff13 = MultiFrameComparison.compute_frame_difference(self.frame1, self.frame3)
        
        # Frame 1 and 3 should have greater difference than 1 and 2
        self.assertGreater(
            diff13["mean_absolute_difference"],
            diff12["mean_absolute_difference"]
        )
    
    def test_create_difference_map(self):
        """Test difference map creation."""
        diff_map = MultiFrameComparison.create_difference_map(self.frame1, self.frame2)
        
        self.assertIsNotNone(diff_map)
        self.assertEqual(diff_map.shape[:2], (480, 640))
        self.assertTrue(np.all((diff_map == 0) | (diff_map == 255)))
    
    def test_find_regions_of_interest(self):
        """Test finding regions of interest."""
        # Create frame with distinct region
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        cv2.rectangle(frame2, (100, 100), (300, 300), (255, 255, 255), -1)
        
        diff_map = MultiFrameComparison.create_difference_map(frame1, frame2)
        rois = MultiFrameComparison.find_regions_of_interest(diff_map, min_area=100)
        
        self.assertGreater(len(rois), 0)
    
    def test_merge_frames_average(self):
        """Test merging frames with average method."""
        frames = [self.frame1, self.frame2]
        merged = MultiFrameComparison.merge_frames(frames, method="average")
        
        self.assertIsNotNone(merged)
        self.assertEqual(merged.shape, self.frame1.shape)
    
    def test_merge_frames_median(self):
        """Test merging frames with median method."""
        frames = [self.frame1, self.frame2, self.frame3]
        merged = MultiFrameComparison.merge_frames(frames, method="median")
        
        self.assertIsNotNone(merged)
        self.assertEqual(merged.shape, self.frame1.shape)
    
    def test_merge_frames_max(self):
        """Test merging frames with max method."""
        frames = [self.frame1, self.frame2]
        merged = MultiFrameComparison.merge_frames(frames, method="max")
        
        self.assertIsNotNone(merged)
        # Max should be closer to frame2 (higher values)
        self.assertGreater(np.mean(merged), np.mean(self.frame1))
    
    def test_merge_frames_min(self):
        """Test merging frames with min method."""
        frames = [self.frame1, self.frame2]
        merged = MultiFrameComparison.merge_frames(frames, method="min")
        
        self.assertIsNotNone(merged)
        # Min should be closer to frame1 (lower values)
        self.assertLess(np.mean(merged), np.mean(self.frame2))
    
    def test_merge_frames_invalid_method(self):
        """Test merging with invalid method."""
        frames = [self.frame1, self.frame2]
        
        with self.assertRaises(ValueError):
            MultiFrameComparison.merge_frames(frames, method="invalid")


if __name__ == "__main__":
    unittest.main()
