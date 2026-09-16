"""
License plate region cropping module.
Allows selection and extraction of plate regions from images.
"""

import logging
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PlateRegion:
    """Represents a license plate region selection."""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        """
        Initialize a plate region.
        
        Args:
            x: X coordinate of top-left corner
            y: Y coordinate of top-left corner
            width: Width of region
            height: Height of region
        """
        self.x = max(0, x)
        self.y = max(0, y)
        self.width = max(1, width)
        self.height = max(1, height)
    
    def get_coordinates(self) -> Tuple[int, int, int, int]:
        """Return region coordinates as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)
    
    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Return region as bounding box (x1, y1, x2, y2)."""
        return (
            self.x,
            self.y,
            self.x + self.width,
            self.y + self.height
        )
    
    def to_dict(self) -> dict:
        """Export region as dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }
    
    @staticmethod
    def from_dict(data: dict) -> "PlateRegion":
        """Create PlateRegion from dictionary."""
        return PlateRegion(
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"]
        )


def crop_plate_region(
    image_path: str | Path,
    region: PlateRegion
) -> np.ndarray:
    """
    Crop a plate region from an image.
    
    Args:
        image_path: Path to image file
        region: PlateRegion specifying the crop area
        
    Returns:
        Cropped image as numpy array
        
    Raises:
        FileNotFoundError: If image not found
        ValueError: If region is invalid
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Validate region
    img_height, img_width = image.shape[:2]
    x1, y1, x2, y2 = region.get_bbox()
    
    # Clamp coordinates to image bounds
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(x1 + 1, min(x2, img_width))
    y2 = max(y1 + 1, min(y2, img_height))
    
    cropped = image[y1:y2, x1:x2]
    
    logger.info(
        f"Plate region cropped: {x1},{y1} to {x2},{y2} "
        f"({x2-x1}x{y2-y1})"
    )
    
    return cropped


def crop_plate_region_from_array(
    image: np.ndarray,
    region: PlateRegion
) -> np.ndarray:
    """
    Crop a plate region from an image array.
    
    Args:
        image: Image as numpy array
        region: PlateRegion specifying the crop area
        
    Returns:
        Cropped image as numpy array
        
    Raises:
        ValueError: If region is invalid
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    img_height, img_width = image.shape[:2]
    x1, y1, x2, y2 = region.get_bbox()
    
    # Clamp coordinates to image bounds
    x1 = max(0, min(x1, img_width - 1))
    y1 = max(0, min(y1, img_height - 1))
    x2 = max(x1 + 1, min(x2, img_width))
    y2 = max(y1 + 1, min(y2, img_height))
    
    cropped = image[y1:y2, x1:x2]
    
    logger.debug(f"Plate region cropped from array: {x2-x1}x{y2-y1}")
    
    return cropped


def validate_region(region: PlateRegion, image_width: int, image_height: int) -> bool:
    """
    Validate that a region is within image bounds.
    
    Args:
        region: PlateRegion to validate
        image_width: Image width
        image_height: Image height
        
    Returns:
        True if region is valid, False otherwise
    """
    x1, y1, x2, y2 = region.get_bbox()
    
    if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        return False
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    return True
