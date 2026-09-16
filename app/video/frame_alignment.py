"""
Frame alignment and registration module.
Aligns multiple frames for multi-frame analysis in forensic investigation.
"""

import logging
from typing import Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameAligner:
    """Aligns frames using feature detection and homography."""
    
    @staticmethod
    def detect_features(image: np.ndarray, max_features: int = 500) -> Tuple[list, np.ndarray]:
        """
        Detect keypoints and descriptors using ORB detector.
        
        Args:
            image: Input image (grayscale)
            max_features: Maximum number of features (default: 500)
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        if image is None or image.size == 0:
            return [], None
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Initialize ORB detector
        orb = cv2.ORB_create(nfeatures=max_features)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        logger.debug(f"Detected {len(keypoints)} features")
        
        return keypoints, descriptors
    
    @staticmethod
    def match_features(
        descriptors1: np.ndarray,
        descriptors2: np.ndarray,
        ratio_test: float = 0.75
    ) -> list:
        """
        Match features between two images using Lowe's ratio test.
        
        Args:
            descriptors1: Descriptors from first image
            descriptors2: Descriptors from second image
            ratio_test: Ratio for Lowe's test (default: 0.75)
            
        Returns:
            List of good matches
        """
        if descriptors1 is None or descriptors2 is None:
            return []
        
        # Create BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Find matches using k-NN
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_test * n.distance:
                    good_matches.append(m)
        
        logger.debug(f"Found {len(good_matches)} good matches")
        
        return good_matches
    
    @staticmethod
    def compute_homography(
        keypoints1: list,
        keypoints2: list,
        matches: list,
        min_matches: int = 4
    ) -> Optional[np.ndarray]:
        """
        Compute homography matrix from matched keypoints.
        
        Args:
            keypoints1: Keypoints from first image
            keypoints2: Keypoints from second image
            matches: Matched keypoints
            min_matches: Minimum matches required (default: 4)
            
        Returns:
            Homography matrix or None if insufficient matches
        """
        if len(matches) < min_matches:
            logger.warning(f"Insufficient matches ({len(matches)} < {min_matches})")
            return None
        
        # Extract matched keypoint coordinates
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Compute homography using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            logger.warning("Failed to compute homography")
            return None
        
        logger.debug("Homography computed successfully")
        
        return H
    
    @staticmethod
    def warp_image(
        image: np.ndarray,
        H: np.ndarray,
        output_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Warp image using homography matrix.
        
        Args:
            image: Input image
            H: Homography matrix
            output_shape: Output image shape (height, width)
            
        Returns:
            Warped image
        """
        if H is None:
            return image.copy()
        
        warped = cv2.warpPerspective(image, H, (output_shape[1], output_shape[0]))
        
        logger.debug(f"Image warped to {output_shape}")
        
        return warped
    
    @staticmethod
    def align_frame_to_reference(
        reference_frame: np.ndarray,
        target_frame: np.ndarray,
        max_features: int = 500
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Align a target frame to a reference frame.
        
        Args:
            reference_frame: Reference frame (BGR or grayscale)
            target_frame: Frame to align (BGR or grayscale)
            max_features: Maximum features to detect (default: 500)
            
        Returns:
            Tuple of (aligned_target_frame, homography_matrix)
        """
        # Convert to grayscale for feature detection
        if len(reference_frame.shape) == 3:
            ref_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = reference_frame
        
        if len(target_frame.shape) == 3:
            tgt_gray = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
        else:
            tgt_gray = target_frame
        
        # Detect features
        kp1, desc1 = FrameAligner.detect_features(ref_gray, max_features)
        kp2, desc2 = FrameAligner.detect_features(tgt_gray, max_features)
        
        if not desc1 is None or not desc2 is None:
            # Match features
            matches = FrameAligner.match_features(desc1, desc2)
            
            # Compute homography
            H = FrameAligner.compute_homography(kp1, kp2, matches)
            
            if H is not None:
                # Warp target frame
                aligned = FrameAligner.warp_image(
                    target_frame,
                    H,
                    (reference_frame.shape[0], reference_frame.shape[1])
                )
                
                logger.info("Frame aligned successfully")
                
                return aligned, H
        
        logger.warning("Failed to align frames")
        
        return None, None


class MultiFrameComparison:
    """Compare and analyze multiple aligned frames."""
    
    @staticmethod
    def compute_frame_difference(
        frame1: np.ndarray,
        frame2: np.ndarray
    ) -> dict:
        """
        Compute difference between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            Dictionary with difference metrics
        """
        if frame1.shape != frame2.shape:
            logger.warning("Frames have different shapes, cannot compute difference")
            return {}
        
        # Convert to grayscale if needed
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1
            gray2 = frame2
        
        # Compute difference
        diff = cv2.absdiff(gray1, gray2)
        
        metrics = {
            "mean_absolute_difference": float(np.mean(diff)),
            "max_difference": float(np.max(diff)),
            "min_difference": float(np.min(diff)),
            "std_deviation": float(np.std(diff)),
            "pixels_different": int(np.count_nonzero(diff > 10))  # Threshold 10
        }
        
        logger.debug(f"Frame difference computed: MAD={metrics['mean_absolute_difference']:.2f}")
        
        return metrics
    
    @staticmethod
    def create_difference_map(
        frame1: np.ndarray,
        frame2: np.ndarray,
        threshold: int = 30
    ) -> np.ndarray:
        """
        Create a difference map showing where frames differ.
        
        Args:
            frame1: First frame
            frame2: Second frame
            threshold: Difference threshold (default: 30)
            
        Returns:
            Binary difference map
        """
        if frame1.shape != frame2.shape:
            return None
        
        # Convert to grayscale
        if len(frame1.shape) == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1
            gray2 = frame2
        
        # Compute difference and threshold
        diff = cv2.absdiff(gray1, gray2)
        _, diff_map = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        logger.debug("Difference map created")
        
        return diff_map
    
    @staticmethod
    def find_regions_of_interest(
        diff_map: np.ndarray,
        min_area: int = 100
    ) -> list:
        """
        Find regions of interest in difference map.
        
        Args:
            diff_map: Binary difference map
            min_area: Minimum area for region (default: 100)
            
        Returns:
            List of (x, y, width, height) bounding boxes
        """
        if diff_map is None:
            return []
        
        # Find contours
        contours, _ = cv2.findContours(diff_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rois = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                rois.append((x, y, w, h))
        
        logger.info(f"Found {len(rois)} regions of interest")
        
        return rois
    
    @staticmethod
    def merge_frames(frames: list, method: str = "average") -> np.ndarray:
        """
        Merge multiple frames using specified method.
        
        Args:
            frames: List of frames (all same shape)
            method: Merge method ("average", "median", "max", "min")
            
        Returns:
            Merged frame
        """
        if not frames:
            return None
        
        frames_array = np.array(frames, dtype=np.float32)
        
        if method == "average":
            merged = np.mean(frames_array, axis=0)
        elif method == "median":
            merged = np.median(frames_array, axis=0)
        elif method == "max":
            merged = np.max(frames_array, axis=0)
        elif method == "min":
            merged = np.min(frames_array, axis=0)
        else:
            raise ValueError(f"Unknown merge method: {method}")
        
        merged = np.clip(merged, 0, 255).astype(np.uint8)
        
        logger.info(f"Merged {len(frames)} frames using {method} method")
        
        return merged
