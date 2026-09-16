"""
Evidence management and intake for forensic analysis.
Handles evidence record creation and preservation.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
import numpy as np

from app.core.hashing import calculate_sha256

logger = logging.getLogger(__name__)

# Supported file formats
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv'}
SUPPORTED_FORMATS = SUPPORTED_IMAGE_FORMATS | SUPPORTED_VIDEO_FORMATS


class Evidence:
    """Represents a piece of forensic evidence (image or video)."""
    
    def __init__(self, file_path: str | Path):
        """
        Initialize an evidence record.
        
        Args:
            file_path: Path to the evidence file
            
        Raises:
            FileNotFoundError: If file does not exist
            ValueError: If file format is not supported
        """
        self.file_path = Path(file_path)
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Evidence file not found: {self.file_path}")
        
        file_ext = self.file_path.suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Generate evidence ID
        self.evidence_id = self._generate_evidence_id()
        
        # Basic file metadata
        self.original_filename = self.file_path.name
        self.file_extension = file_ext
        self.file_size = self.file_path.stat().st_size
        self.sha256 = calculate_sha256(self.file_path)
        self.intake_timestamp = datetime.now()
        
        # Determine type
        self.evidence_type = self._determine_type()
        
        # Load type-specific metadata
        if self.evidence_type == "image":
            self._load_image_metadata()
        elif self.evidence_type == "video":
            self._load_video_metadata()
        
        logger.info(f"Evidence loaded: {self.evidence_id} - {self.original_filename}")
    
    @staticmethod
    def _generate_evidence_id() -> str:
        """Generate a unique evidence ID."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        import random
        random_part = random.randint(1000, 9999)
        return f"EV-{timestamp}-{random_part}"
    
    def _determine_type(self) -> str:
        """Determine if evidence is image or video."""
        ext = self.file_extension.lower()
        if ext in SUPPORTED_IMAGE_FORMATS:
            return "image"
        elif ext in SUPPORTED_VIDEO_FORMATS:
            return "video"
        else:
            raise ValueError(f"Unknown file type: {ext}")
    
    def _load_image_metadata(self) -> None:
        """Load metadata for image evidence."""
        try:
            img = cv2.imread(str(self.file_path))
            if img is None:
                raise ValueError("Could not read image")
            
            self.height, self.width = img.shape[:2]
            self.channels = img.shape[2] if len(img.shape) > 2 else 1
            self.resolution = f"{self.width}x{self.height}"
            
            # Additional metadata
            self.fps = None
            self.frame_count = None
            self.duration = None
            self.codec = None
            
            logger.info(f"Image metadata: {self.resolution}, {self.channels} channels")
            
        except Exception as e:
            logger.error(f"Failed to load image metadata: {e}")
            raise
    
    def _load_video_metadata(self) -> None:
        """Load metadata for video evidence."""
        try:
            cap = cv2.VideoCapture(str(self.file_path))
            
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.resolution = f"{self.width}x{self.height}"
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.duration = self.frame_count / self.fps if self.fps > 0 else 0
            
            # Codec information
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            self.codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            self.channels = None  # Videos are typically multi-channel
            
            cap.release()
            
            logger.info(
                f"Video metadata: {self.resolution}, {self.fps:.2f} FPS, "
                f"{self.frame_count} frames, {self.duration:.2f}s"
            )
            
        except Exception as e:
            logger.error(f"Failed to load video metadata: {e}")
            raise
    
    def to_dict(self) -> dict:
        """Export evidence metadata as dictionary."""
        return {
            "evidence_id": self.evidence_id,
            "original_filename": self.original_filename,
            "file_extension": self.file_extension,
            "file_size": self.file_size,
            "sha256": self.sha256,
            "type": self.evidence_type,
            "resolution": self.resolution,
            "intake_timestamp": self.intake_timestamp.isoformat(),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "codec": self.codec,
            "software_version": "1.0.0"
        }
