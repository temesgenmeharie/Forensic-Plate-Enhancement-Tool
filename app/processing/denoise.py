"""
Noise reduction and denoising module.
Implements multiple denoising techniques for forensic image enhancement.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def apply_bilateral_filter(
    image: np.ndarray,
    diameter: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0
) -> np.ndarray:
    """
    Apply bilateral filtering for edge-preserving denoising.
    
    Bilateral filtering is particularly useful for license plates because:
    - Preserves sharp edges (important for character recognition)
    - Reduces noise in flat regions
    - Maintains local structure
    
    Args:
        image: Input image (BGR or grayscale)
        diameter: Diameter of pixel neighborhood (default: 9)
        sigma_color: Color space standard deviation (default: 75.0)
        sigma_space: Coordinate space standard deviation (default: 75.0)
        
    Returns:
        Denoised image
        
    Raises:
        ValueError: If image is invalid or parameters are invalid
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    if diameter <= 0 or diameter % 2 == 0:
        raise ValueError(f"Diameter must be positive odd number, got {diameter}")
    
    if sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("Sigma values must be positive")
    
    # Apply bilateral filter
    if len(image.shape) == 3:
        filtered = cv2.bilateralFilter(
            image,
            d=diameter,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space
        )
    else:
        # Convert grayscale to BGR for filtering, then back
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        filtered_bgr = cv2.bilateralFilter(
            bgr,
            d=diameter,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space
        )
        filtered = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2GRAY)
    
    logger.info(
        f"Bilateral filter applied: d={diameter}, "
        f"sigma_color={sigma_color}, sigma_space={sigma_space}"
    )
    
    return filtered


def apply_non_local_means_denoise(
    image: np.ndarray,
    h: float = 10.0,
    template_window_size: int = 7,
    search_window_size: int = 21
) -> np.ndarray:
    """
    Apply Non-Local Means (NLM) denoising.
    
    NLM is more sophisticated than bilateral filtering and often produces
    better results, but is slower.
    
    Args:
        image: Input image (BGR or grayscale)
        h: Filter strength (higher = more filtering but more blur)
        template_window_size: Size of template patches (default: 7)
        search_window_size: Size of search area (default: 21)
        
    Returns:
        Denoised image
        
    Raises:
        ValueError: If image is invalid or parameters are invalid
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    if h <= 0:
        raise ValueError(f"h must be positive, got {h}")
    
    if template_window_size <= 0 or template_window_size % 2 == 0:
        raise ValueError(f"Template window size must be positive odd, got {template_window_size}")
    
    if search_window_size <= 0 or search_window_size % 2 == 0:
        raise ValueError(f"Search window size must be positive odd, got {search_window_size}")
    
    # Apply NLM denoising
    if len(image.shape) == 3:
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            h=h,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )
    else:
        denoised = cv2.fastNlMeansDenoising(
            image,
            h=h,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )
    
    logger.info(
        f"Non-Local Means denoising applied: h={h}, "
        f"template={template_window_size}, search={search_window_size}"
    )
    
    return denoised


def apply_morphological_opening(
    image: np.ndarray,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Apply morphological opening (erosion followed by dilation).
    
    Useful for removing small noise while preserving larger structures.
    
    Args:
        image: Input image (grayscale)
        kernel_size: Size of morphological kernel (default: 5)
        
    Returns:
        Processed image
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"Kernel size must be positive odd, got {kernel_size}")
    
    # Create kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Apply morphological opening
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    
    logger.info(f"Morphological opening applied: kernel={kernel_size}")
    
    return opened


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian blur for noise reduction.
    
    Simple but effective technique. Use with caution on license plates
    to avoid excessive blur of characters.
    
    Args:
        image: Input image
        kernel_size: Size of Gaussian kernel (default: 5, must be odd)
        sigma: Standard deviation of Gaussian kernel (default: 1.0)
        
    Returns:
        Blurred image
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"Kernel size must be positive odd, got {kernel_size}")
    
    if sigma <= 0:
        raise ValueError(f"Sigma must be positive, got {sigma}")
    
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    logger.info(f"Gaussian blur applied: kernel={kernel_size}, sigma={sigma}")
    
    return blurred


def calculate_noise_metrics(image: np.ndarray) -> dict:
    """
    Estimate noise level in image using Laplacian variance.
    
    Higher variance = sharper/less noisy image.
    Lower variance = blurrier/noisier image.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        Dictionary with noise metrics
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian variance (sharpness metric)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = laplacian.var()
    
    # Calculate standard deviation as additional metric
    std_dev = float(np.std(gray))
    
    metrics = {
        "laplacian_variance": float(laplacian_var),
        "std_deviation": std_dev,
        "is_sharp": laplacian_var > 100  # Threshold for sharpness
    }
    
    logger.debug(f"Noise metrics: {metrics}")
    
    return metrics
