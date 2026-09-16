"""
Contrast enhancement module.
Implements CLAHE (Contrast Limited Adaptive Histogram Equalization) for forensic analysis.
"""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    CLAHE is particularly effective for license plate enhancement as it:
    - Preserves local contrast
    - Prevents over-enhancement in dark/bright regions
    - Reduces noise amplification
    
    Args:
        image: Input image (BGR or grayscale)
        clip_limit: Threshold for contrast limiting (default: 2.0)
        tile_grid_size: Size of grid tiles for adaptive processing (default: 8x8)
        
    Returns:
        Enhanced image with CLAHE applied
        
    Raises:
        ValueError: If image is invalid or parameters are invalid
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    if clip_limit <= 0:
        raise ValueError(f"Clip limit must be positive, got {clip_limit}")
    
    if tile_grid_size[0] <= 0 or tile_grid_size[1] <= 0:
        raise ValueError(f"Tile grid size must be positive, got {tile_grid_size}")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create CLAHE object
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    
    # Apply CLAHE
    enhanced = clahe.apply(gray)
    
    logger.info(
        f"CLAHE applied: clip_limit={clip_limit}, "
        f"tile_grid={tile_grid_size}"
    )
    
    return enhanced


def apply_clahe_color(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE to color image (convert to LAB, apply to L channel, convert back).
    
    This preserves color information while enhancing contrast.
    
    Args:
        image: Input BGR image
        clip_limit: Threshold for contrast limiting (default: 2.0)
        tile_grid_size: Size of grid tiles for adaptive processing (default: 8x8)
        
    Returns:
        Enhanced BGR image with CLAHE applied to lightness
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel color image")
    
    # Convert BGR to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split into channels
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    l_enhanced = clahe.apply(l_channel)
    
    # Merge channels back
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    logger.info(
        f"CLAHE (color-preserving) applied: clip_limit={clip_limit}, "
        f"tile_grid={tile_grid_size}"
    )
    
    return enhanced


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Apply standard histogram equalization (not adaptive).
    
    This is less sophisticated than CLAHE but may be useful for comparison.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        Image with histogram equalization applied
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)
    
    logger.info("Histogram equalization applied")
    
    return equalized


def calculate_contrast_metrics(image: np.ndarray) -> dict:
    """
    Calculate contrast-related metrics for image quality assessment.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        Dictionary with contrast metrics
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate metrics
    metrics = {
        "mean": float(np.mean(gray)),
        "std_dev": float(np.std(gray)),
        "min": int(np.min(gray)),
        "max": int(np.max(gray)),
        "dynamic_range": int(np.max(gray) - np.min(gray))
    }
    
    logger.debug(f"Contrast metrics calculated: {metrics}")
    
    return metrics
