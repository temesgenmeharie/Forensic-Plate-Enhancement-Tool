"""
Sharpening and edge enhancement module.
Implements unsharp masking and other sharpening techniques.
"""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def apply_unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.0,
    amount: float = 1.5,
    threshold: float = 0.0
) -> np.ndarray:
    """
    Apply unsharp masking for sharpening.
    
    Unsharp masking works by:
    1. Creating a blurred version of the image
    2. Subtracting the blurred version from the original
    3. Adding the result back to the original (scaled by amount)
    
    This technique preserves the original while enhancing edges.
    
    Args:
        image: Input image (BGR or grayscale)
        sigma: Gaussian blur sigma for blurred version (default: 1.0)
        amount: Strength of sharpening effect (default: 1.5, typically 0.5-2.0)
        threshold: Minimum gradient to sharpen (default: 0.0, range 0-255)
        
    Returns:
        Sharpened image
        
    Raises:
        ValueError: If image is invalid or parameters are invalid
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")
    
    if amount < 0:
        raise ValueError(f"Amount must be non-negative, got {amount}")
    
    if threshold < 0 or threshold > 255:
        raise ValueError(f"Threshold must be 0-255, got {threshold}")
    
    # Convert to float for processing
    image_float = image.astype(np.float32) / 255.0
    
    # Create Gaussian blur
    kernel_size = int(2 * np.ceil(3 * sigma)) + 1
    blurred = cv2.GaussianBlur(image_float, (kernel_size, kernel_size), sigma)
    
    # Calculate unsharp mask
    mask = image_float - blurred
    
    # Apply threshold if specified
    if threshold > 0:
        threshold_norm = threshold / 255.0
        mask = np.where(np.abs(mask) >= threshold_norm, mask, 0)
    
    # Combine with original
    sharpened = image_float + amount * mask
    
    # Clip to valid range and convert back
    sharpened = np.clip(sharpened * 255, 0, 255).astype(image.dtype)
    
    logger.info(
        f"Unsharp mask applied: sigma={sigma}, amount={amount}, threshold={threshold}"
    )
    
    return sharpened


def apply_laplacian_sharpening(
    image: np.ndarray,
    strength: float = 1.0
) -> np.ndarray:
    """
    Apply Laplacian-based sharpening.
    
    Uses Laplacian edge detection to create a sharpening mask.
    
    Args:
        image: Input image (BGR or grayscale)
        strength: Strength of sharpening (default: 1.0, typically 0.5-2.0)
        
    Returns:
        Sharpened image
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    if strength < 0:
        raise ValueError(f"Strength must be non-negative, got {strength}")
    
    # Convert to float
    image_float = image.astype(np.float32)
    
    # Calculate Laplacian
    laplacian = cv2.Laplacian(image_float, cv2.CV_32F)
    
    # Apply sharpening
    sharpened = image_float - strength * laplacian
    
    # Clip and convert back
    sharpened = np.clip(sharpened, 0, 255).astype(image.dtype)
    
    logger.info(f"Laplacian sharpening applied: strength={strength}")
    
    return sharpened


def apply_high_pass_filter(
    image: np.ndarray,
    sigma: float = 2.0,
    amount: float = 1.0
) -> np.ndarray:
    """
    Apply high-pass filter for edge enhancement.
    
    Args:
        image: Input image
        sigma: Gaussian blur sigma (default: 2.0)
        amount: Enhancement strength (default: 1.0)
        
    Returns:
        Enhanced image with high-pass filter applied
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    if sigma <= 0 or amount < 0:
        raise ValueError("Sigma must be positive, amount non-negative")
    
    # Convert to float
    image_float = image.astype(np.float32) / 255.0
    
    # Create Gaussian blur
    kernel_size = int(2 * np.ceil(3 * sigma)) + 1
    blurred = cv2.GaussianBlur(image_float, (kernel_size, kernel_size), sigma)
    
    # High-pass filter = original - blurred
    high_pass = image_float - blurred
    
    # Combine with original
    enhanced = image_float + amount * high_pass
    
    # Clip and convert back
    enhanced = np.clip(enhanced * 255, 0, 255).astype(image.dtype)
    
    logger.info(f"High-pass filter applied: sigma={sigma}, amount={amount}")
    
    return enhanced


def apply_kernel_sharpening(
    image: np.ndarray,
    kernel_type: str = "sharp"
) -> np.ndarray:
    """
    Apply convolution-based sharpening with predefined kernels.
    
    Args:
        image: Input image
        kernel_type: Type of kernel ("sharp", "stronger", "edge_detect")
        
    Returns:
        Sharpened image
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    # Define kernels
    kernels = {
        "sharp": np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ]) / 1.0,
        "stronger": np.array([
            [-2, -1,  0],
            [-1,  1,  1],
            [ 0,  1,  2]
        ]) / 1.0,
        "edge_detect": np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]) / 1.0
    }
    
    if kernel_type not in kernels:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    kernel = kernels[kernel_type]
    
    # Apply convolution
    sharpened = cv2.filter2D(image, -1, kernel)
    
    # Clip to valid range
    sharpened = np.clip(sharpened, 0, 255).astype(image.dtype)
    
    logger.info(f"Kernel sharpening applied: kernel_type={kernel_type}")
    
    return sharpened


def calculate_sharpness_metrics(image: np.ndarray) -> dict:
    """
    Calculate sharpness metrics for image quality assessment.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        Dictionary with sharpness metrics
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate various sharpness metrics
    # 1. Laplacian variance (higher = sharper)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_variance = laplacian.var()
    
    # 2. Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_variance = (sobelx ** 2 + sobely ** 2).mean()
    
    # 3. Gradient magnitude
    gradient = np.sqrt(sobelx ** 2 + sobely ** 2)
    gradient_mean = gradient.mean()
    
    metrics = {
        "laplacian_variance": float(laplacian_variance),
        "sobel_variance": float(sobel_variance),
        "gradient_mean": float(gradient_mean),
        "is_sharp": laplacian_variance > 100
    }
    
    logger.debug(f"Sharpness metrics: {metrics}")
    
    return metrics
