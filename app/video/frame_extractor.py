"""
Frame extraction and processing module for video forensic analysis.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2

from app.video.reader import VideoReader
from app.video.frame_quality import FrameQualityMetrics, FrameSelector

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extract and analyze frames from video files."""
    
    def __init__(self, video_path: str | Path, output_dir: Optional[str | Path] = None):
        """
        Initialize frame extractor.
        
        Args:
            video_path: Path to video file
            output_dir: Directory for output frames (optional)
        """
        self.video_reader = VideoReader(video_path)
        self.output_dir = Path(output_dir) if output_dir else None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Frame extractor initialized with output dir: {self.output_dir}")
        else:
            logger.info("Frame extractor initialized without output directory")
    
    def extract_keyframes(
        self,
        interval: float = 1.0,
        start_time: float = 0.0,
        end_time: Optional[float] = None
    ) -> list:
        """
        Extract keyframes at regular intervals.
        
        Args:
            interval: Interval between frames in seconds (default: 1.0)
            start_time: Start time in seconds (default: 0.0)
            end_time: End time in seconds (default: video duration)
            
        Returns:
            List of (frame_number, frame_image, metrics) tuples
        """
        frames_data = self.video_reader.read_time_range(start_time, end_time, interval)
        
        frames_with_metrics = []
        
        for frame_num, frame in frames_data:
            metrics = FrameQualityMetrics.calculate_all_metrics(frame)
            frames_with_metrics.append((frame_num, frame, metrics))
        
        logger.info(f"Extracted {len(frames_with_metrics)} keyframes")
        
        return frames_with_metrics
    
    def extract_all_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        stride: int = 1
    ) -> list:
        """
        Extract all frames (or sample with stride).
        
        Args:
            start_frame: Start frame number (default: 0)
            end_frame: End frame number (default: last frame)
            stride: Extract every nth frame (default: 1)
            
        Returns:
            List of (frame_number, frame_image, metrics) tuples
        """
        frames_data = self.video_reader.read_frame_range(start_frame, end_frame, stride)
        
        frames_with_metrics = []
        
        for frame_num, frame in frames_data:
            metrics = FrameQualityMetrics.calculate_all_metrics(frame)
            frames_with_metrics.append((frame_num, frame, metrics))
        
        logger.info(f"Extracted {len(frames_with_metrics)} frames")
        
        return frames_with_metrics
    
    def select_best_frames(
        self,
        frames_with_metrics: list,
        selection_method: str = "composite",
        count: int = 5
    ) -> list:
        """
        Select best frames using specified method.
        
        Args:
            frames_with_metrics: List of (frame_number, frame_image, metrics) tuples
            selection_method: Method to use ("sharpness", "brightness_range", "contrast", "composite")
            count: Number of frames to select
            
        Returns:
            List of (frame_number, frame_image) tuples
        """
        selector = FrameSelector(frames_with_metrics)
        
        if selection_method == "sharpness":
            selected = selector.select_by_sharpness(top_n=count)
        elif selection_method == "contrast":
            selected = selector.select_by_contrast(threshold=15)
            selected = selected[:count]
        elif selection_method == "brightness_range":
            selected = selector.select_by_brightness_range()
            selected = selected[:count]
        elif selection_method == "composite":
            selected = selector.select_best_composite(top_n=count)
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
        
        logger.info(f"Selected {len(selected)} frames using {selection_method} method")
        
        return selected
    
    def save_frames(
        self,
        frames: list,
        prefix: str = "frame"
    ) -> list:
        """
        Save extracted frames to disk.
        
        Args:
            frames: List of (frame_number, frame_image) tuples
            prefix: Filename prefix (default: "frame")
            
        Returns:
            List of saved file paths
        """
        if not self.output_dir:
            logger.warning("Output directory not set, cannot save frames")
            return []
        
        saved_paths = []
        
        for frame_num, frame in frames:
            filename = f"{prefix}_{frame_num:06d}.png"
            filepath = self.output_dir / filename
            
            success = cv2.imwrite(str(filepath), frame)
            
            if success:
                saved_paths.append(filepath)
                logger.debug(f"Saved frame {frame_num} to {filepath}")
            else:
                logger.error(f"Failed to save frame {frame_num}")
        
        logger.info(f"Saved {len(saved_paths)} frames to {self.output_dir}")
        
        return saved_paths
    
    def get_video_metadata(self) -> dict:
        """Get video metadata."""
        return self.video_reader.get_metadata()
    
    def close(self):
        """Close video file."""
        self.video_reader.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
