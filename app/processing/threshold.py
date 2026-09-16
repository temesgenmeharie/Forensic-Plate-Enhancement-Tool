"""
Thresholding and binarization module.
Implements adaptive thresholding for character detection and analysis.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def apply_adaptive_gaussian_threshold(
    image: np.ndarray,
    block_size: int = 11,
    constant: float = 2.0,
    max_value: int = 255
) -> np.ndarray:
    """
    Apply Gaussian adaptive thresholding.
    
    Adaptive thresholding calculates threshold for each pixel based on
    local neighborhood statistics. Gaussian variant uses weighted Gaussian
    window for neighborhood calculation.
    
    Useful for license plates because:
    - Handles variable lighting conditions
    - Preserves local contrast
    - Better for text detection
    
    Args:
        image: Input image (grayscale or BGR)
        block_size: Size of neighborhood area (default: 11, must be odd)
        constant: Constant subtracted from mean (default: 2.0)
        max_value: Maximum value assigned to pixels (default: 255)
        
    Returns:
        Binary image (0 or 255)
        
    Raises:
        ValueError: If image is invalid or parameters are invalid
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    if block_size <= 0 or block_size % 2 == 0:
        raise ValueError(f"Block size must be positive odd, got {block_size}")
    
    if max_value < 0 or max_value > 255:
        raise ValueError(f"Max value must be 0-255, got {max_value}")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply adaptive Gaussian threshold
    thresholded = cv2.adaptiveThreshold(
        gray,
        maxValue=max_value,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=constant
    )
    
    logger.info(
        f"Adaptive Gaussian threshold applied: "
        f"block_size={block_size}, constant={constant}"
    )
    
    return thresholded


def apply_adaptive_mean_threshold(
    image: np.ndarray,
    block_size: int = 11,
    constant: float = 2.0,
    max_value: int = 255
) -> np.ndarray:
    """
    Apply mean adaptive thresholding.
    
    Mean adaptive threshold uses simple mean of neighborhood rather than
    weighted Gaussian mean. Faster but potentially less effective.
    
    Args:
        image: Input image (grayscale or BGR)
        block_size: Size of neighborhood area (default: 11, must be odd)
        constant: Constant subtracted from mean (default: 2.0)
        max_value: Maximum value assigned to pixels (default: 255)
        
    Returns:
        Binary image (0 or 255)
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    if block_size <= 0 or block_size % 2 == 0:
        raise ValueError(f"Block size must be positive odd, got {block_size}")
    
    if max_value < 0 or max_value > 255:
        raise ValueError(f"Max value must be 0-255, got {max_value}")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply adaptive mean threshold
    thresholded = cv2.adaptiveThreshold(
        gray,
        maxValue=max_value,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=constant
    )
    
    logger.info(
        f"Adaptive mean threshold applied: "
        f"block_size={block_size}, constant={constant}"
    )
    
    return thresholded


def apply_otsu_threshold(
    image: np.ndarray,
    max_value: int = 255
) -> tuple[np.ndarray, float]:
    """
    Apply Otsu's automatic thresholding.
    
    Otsu's method automatically calculates optimal threshold value
    by minimizing within-class variance.
    
    Args:
        image: Input image (grayscale or BGR)
        max_value: Maximum value assigned to pixels (default: 255)
        
    Returns:
        Tuple of (binary image, threshold value used)
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Otsu's thresholding
    threshold_value, thresholded = cv2.threshold(
        gray,
        0,
        max_value,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    logger.info(f"Otsu threshold applied: threshold_value={threshold_value:.1f}")
    
    return thresholded, threshold_value


def apply_binary_threshold(
    image: np.ndarray,
    threshold_value: int = 127,
    max_value: int = 255
) -> np.ndarray:
    """
    Apply simple binary thresholding with fixed threshold.
    
    Args:
        image: Input image (grayscale or BGR)
        threshold_value: Threshold value (default: 127)
        max_value: Maximum value assigned to pixels (default: 255)
        
    Returns:
        Binary image
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    if threshold_value < 0 or threshold_value > 255:
        raise ValueError(f"Threshold value must be 0-255, got {threshold_value}")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply binary threshold
    _, thresholded = cv2.threshold(
        gray,
        threshold_value,
        max_value,
        cv2.THRESH_BINARY
    )
    
    logger.info(f"Binary threshold applied: threshold={threshold_value}")
    
    return thresholded


def apply_inverse_binary_threshold(
    image: np.ndarray,
    threshold_value: int = 127,
    max_value: int = 255
) -> np.ndarray:
    """
    Apply inverse binary thresholding (dark becomes white, light becomes black).
    
    Useful for plates with white characters on dark background.
    
    Args:
        image: Input image (grayscale or BGR)
        threshold_value: Threshold value (default: 127)
        max_value: Maximum value assigned to pixels (default: 255)
        
    Returns:
        Binary image (inverted)
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply inverse binary threshold
    _, thresholded = cv2.threshold(
        gray,
        threshold_value,
        max_value,
        cv2.THRESH_BINARY_INV
    )
    
    logger.info(f"Inverse binary threshold applied: threshold={threshold_value}")
    
    return thresholded


def calculate_threshold_metrics(image: np.ndarray) -> dict:
    """
    Calculate metrics to help choose optimal threshold.
    
    Args:
        image: Input image (grayscale or BGR)
        
    Returns:
        Dictionary with threshold analysis metrics
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    # Calculate Otsu's threshold
    _, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    
    # Calculate dynamic range
    min_val = gray.min()
    max_val = gray.max()
    
    metrics = {
        "min_value": int(min_val),
        "max_value": int(max_val),
        "dynamic_range": int(max_val - min_val),
        "mean": float(gray.mean()),
        "std_dev": float(gray.std()),
        "histogram_peaks": int(np.sum(hist > 0.01))  # Count significant peaks
    }
    
    logger.debug(f"Threshold metrics: {metrics}")
    
    return metrics
