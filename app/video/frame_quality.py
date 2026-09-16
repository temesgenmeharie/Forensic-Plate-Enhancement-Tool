"""
Frame quality assessment module.
Evaluates image quality metrics for frame selection in forensic analysis.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameQualityMetrics:
    """Calculate and track frame quality metrics."""
    
    @staticmethod
    def calculate_sharpness(frame: np.ndarray) -> float:
        """
        Calculate sharpness metric using Laplacian variance.
        
        Higher values indicate sharper images.
        
        Args:
            frame: Frame image
            
        Returns:
            Sharpness score (typically 0-500, higher is sharper)
        """
        if frame is None or frame.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        return float(sharpness)
    
    @staticmethod
    def calculate_brightness(frame: np.ndarray) -> float:
        """
        Calculate average brightness of frame.
        
        Args:
            frame: Frame image
            
        Returns:
            Brightness value (0-255)
        """
        if frame is None or frame.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        brightness = float(np.mean(gray))
        
        return brightness
    
    @staticmethod
    def calculate_contrast(frame: np.ndarray) -> float:
        """
        Calculate contrast metric using standard deviation.
        
        Higher values indicate more contrast.
        
        Args:
            frame: Frame image
            
        Returns:
            Contrast value (0-255, higher is more contrast)
        """
        if frame is None or frame.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        contrast = float(np.std(gray))
        
        return contrast
    
    @staticmethod
    def calculate_edge_density(frame: np.ndarray) -> float:
        """
        Calculate edge density using Sobel operator.
        
        Higher values indicate more edges/detail.
        
        Args:
            frame: Frame image
            
        Returns:
            Edge density (0-1, higher indicates more edges)
        """
        if frame is None or frame.size == 0:
            return 0.0
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Calculate Sobel edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        edge_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        
        # Normalize by image size and bit depth
        max_possible = 255 * np.sqrt(2) * gray.size
        edge_density = float(edge_magnitude.sum() / max_possible)
        
        return min(1.0, edge_density)
    
    @staticmethod
    def calculate_blur_score(frame: np.ndarray) -> float:
        """
        Calculate blur score (inverse of sharpness).
        
        Lower values indicate less blur (sharper).
        Higher values indicate more blur.
        
        Args:
            frame: Frame image
            
        Returns:
            Blur score (0-100, lower is sharper)
        """
        sharpness = FrameQualityMetrics.calculate_sharpness(frame)
        
        # Convert to 0-100 scale (inverted)
        # Assuming typical sharpness values 0-500
        blur_score = max(0, min(100, 100 * (1 - sharpness / 500)))
        
        return blur_score
    
    @staticmethod
    def calculate_saturation(frame: np.ndarray) -> float:
        """
        Calculate color saturation (only for color frames).
        
        Higher values indicate more saturated colors.
        
        Args:
            frame: Frame image
            
        Returns:
            Saturation value (0-255, higher is more saturated)
        """
        if frame is None or frame.size == 0:
            return 0.0
        
        if len(frame.shape) != 3:
            return 0.0  # Grayscale has no saturation
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Extract saturation channel
        saturation = hsv[:, :, 1]
        
        # Calculate mean saturation
        mean_saturation = float(np.mean(saturation))
        
        return mean_saturation
    
    @staticmethod
    def calculate_all_metrics(frame: np.ndarray) -> dict:
        """
        Calculate all quality metrics for a frame.
        
        Args:
            frame: Frame image
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {
            "sharpness": FrameQualityMetrics.calculate_sharpness(frame),
            "brightness": FrameQualityMetrics.calculate_brightness(frame),
            "contrast": FrameQualityMetrics.calculate_contrast(frame),
            "edge_density": FrameQualityMetrics.calculate_edge_density(frame),
            "blur_score": FrameQualityMetrics.calculate_blur_score(frame),
            "saturation": FrameQualityMetrics.calculate_saturation(frame)
        }
        
        return metrics


class FrameSelector:
    """Select best frames based on quality metrics."""
    
    def __init__(self, frames_with_metrics: list):
        """
        Initialize frame selector.
        
        Args:
            frames_with_metrics: List of (frame_number, frame_image, metrics) tuples
        """
        self.frames_with_metrics = frames_with_metrics
    
    def select_by_sharpness(self, top_n: int = 5) -> list:
        """
        Select frames with highest sharpness.
        
        Args:
            top_n: Number of frames to select (default: 5)
            
        Returns:
            List of (frame_number, frame_image) tuples
        """
        sorted_frames = sorted(
            self.frames_with_metrics,
            key=lambda x: x[2]["sharpness"],
            reverse=True
        )
        
        selected = [(f[0], f[1]) for f in sorted_frames[:top_n]]
        
        logger.info(f"Selected {len(selected)} sharpest frames")
        
        return selected
    
    def select_by_brightness_range(
        self,
        min_brightness: float = 50,
        max_brightness: float = 200
    ) -> list:
        """
        Select frames within brightness range.
        
        Args:
            min_brightness: Minimum brightness (default: 50)
            max_brightness: Maximum brightness (default: 200)
            
        Returns:
            List of (frame_number, frame_image) tuples
        """
        selected = [
            (f[0], f[1]) for f in self.frames_with_metrics
            if min_brightness <= f[2]["brightness"] <= max_brightness
        ]
        
        logger.info(f"Selected {len(selected)} frames in brightness range {min_brightness}-{max_brightness}")
        
        return selected
    
    def select_by_contrast(self, threshold: float = 20) -> list:
        """
        Select frames with sufficient contrast.
        
        Args:
            threshold: Minimum contrast (default: 20)
            
        Returns:
            List of (frame_number, frame_image) tuples
        """
        selected = [
            (f[0], f[1]) for f in self.frames_with_metrics
            if f[2]["contrast"] >= threshold
        ]
        
        logger.info(f"Selected {len(selected)} frames with contrast >= {threshold}")
        
        return selected
    
    def select_best_composite(self, top_n: int = 5) -> list:
        """
        Select frames based on composite quality score.
        
        Combines multiple metrics: sharpness, contrast, brightness, edge density.
        
        Args:
            top_n: Number of frames to select (default: 5)
            
        Returns:
            List of (frame_number, frame_image) tuples
        """
        # Calculate composite scores
        composite_frames = []
        
        for frame_num, frame, metrics in self.frames_with_metrics:
            # Normalize metrics to 0-1 scale
            sharpness_norm = min(1.0, metrics["sharpness"] / 500)
            contrast_norm = min(1.0, metrics["contrast"] / 100)
            brightness_norm = 1.0 - abs(metrics["brightness"] - 128) / 128
            edge_norm = metrics["edge_density"]
            
            # Composite score (weighted average)
            composite_score = (
                sharpness_norm * 0.4 +
                contrast_norm * 0.3 +
                brightness_norm * 0.2 +
                edge_norm * 0.1
            )
            
            composite_frames.append((frame_num, frame, composite_score))
        
        # Sort by composite score
        sorted_frames = sorted(composite_frames, key=lambda x: x[2], reverse=True)
        
        selected = [(f[0], f[1]) for f in sorted_frames[:top_n]]
        
        logger.info(f"Selected {len(selected)} best frames by composite quality")
        
        return selected
