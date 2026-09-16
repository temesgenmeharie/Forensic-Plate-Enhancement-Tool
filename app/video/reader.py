"""
Video reading and metadata extraction module.
Handles video file operations for forensic analysis.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


class VideoReader:
    """Handles reading and metadata extraction from video files."""
    
    def __init__(self, video_path: str | Path):
        """
        Initialize video reader.
        
        Args:
            video_path: Path to video file
            
        Raises:
            FileNotFoundError: If video file not found
            ValueError: If video cannot be opened
        """
        self.video_path = Path(video_path)
        self.capture = None  # Initialize to None for safety
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        # Open video
        self.capture = cv2.VideoCapture(str(self.video_path))
        
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Extract metadata
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        # Codec information
        fourcc = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        self.codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        logger.info(
            f"Video opened: {self.width}x{self.height}, {self.fps:.2f} FPS, "
            f"{self.frame_count} frames, {self.duration:.2f}s, codec: {self.codec}"
        )
    
    def get_metadata(self) -> dict:
        """
        Get video metadata.
        
        Returns:
            Dictionary with video metadata
        """
        return {
            "filename": self.video_path.name,
            "path": str(self.video_path),
            "width": self.width,
            "height": self.height,
            "resolution": f"{self.width}x{self.height}",
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "codec": self.codec
        }
    
    def read_frame(self, frame_number: Optional[int] = None) -> Tuple[bool, any]:
        """
        Read a specific frame or next frame.
        
        Args:
            frame_number: Frame number to read (0-indexed), or None for next frame
            
        Returns:
            Tuple of (success, frame_image)
        """
        if frame_number is not None:
            if frame_number < 0 or frame_number >= self.frame_count:
                logger.warning(f"Frame number out of range: {frame_number}")
                return False, None
            
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        success, frame = self.capture.read()
        
        if success:
            logger.debug(f"Frame read: {int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1}")
        
        return success, frame
    
    def read_frame_range(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1
    ) -> list:
        """
        Read a range of frames.
        
        Args:
            start_frame: Start frame number (default: 0)
            end_frame: End frame number inclusive (default: last frame)
            step: Read every nth frame (default: 1)
            
        Returns:
            List of frame images
        """
        if start_frame < 0:
            start_frame = 0
        
        if end_frame is None or end_frame >= self.frame_count:
            end_frame = self.frame_count - 1
        
        if step <= 0:
            raise ValueError(f"Step must be positive, got {step}")
        
        frames = []
        
        for frame_num in range(start_frame, end_frame + 1, step):
            success, frame = self.read_frame(frame_num)
            
            if success:
                frames.append((frame_num, frame))
            else:
                logger.warning(f"Failed to read frame {frame_num}")
        
        logger.info(f"Read {len(frames)} frames from {start_frame} to {end_frame} (step {step})")
        
        return frames
    
    def read_time_range(
        self,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        interval: float = 0.5
    ) -> list:
        """
        Read frames within a time range.
        
        Args:
            start_time: Start time in seconds (default: 0.0)
            end_time: End time in seconds (default: video duration)
            interval: Interval between frames in seconds (default: 0.5)
            
        Returns:
            List of (frame_number, frame_image) tuples
        """
        if start_time < 0:
            start_time = 0.0
        
        if end_time is None or end_time > self.duration:
            end_time = self.duration
        
        if interval <= 0:
            raise ValueError(f"Interval must be positive, got {interval}")
        
        # Convert time to frame numbers
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        step = max(1, int(interval * self.fps))
        
        frames = self.read_frame_range(start_frame, end_frame, step)
        
        logger.info(
            f"Read {len(frames)} frames from {start_time:.2f}s to {end_time:.2f}s "
            f"(interval {interval}s)"
        )
        
        return frames
    
    def close(self) -> None:
        """Close video file."""
        if self.capture is not None:
            self.capture.release()
            logger.info("Video reader closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor."""
        self.close()
