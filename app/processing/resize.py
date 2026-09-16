"""
Image upscaling and resizing module.
Implements multiple interpolation methods for license plate enhancement.
"""

import logging
from enum import Enum
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class InterpolationMethod(Enum):
    """Supported interpolation methods."""
    NEAREST = cv2.INTER_NEAREST
    LINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    LANCZOS4 = cv2.INTER_LANCZOS4
    AREA = cv2.INTER_AREA


def get_interpolation_method(method_name: str) -> int:
    """
    Get OpenCV interpolation constant from method name.
    
    Args:
        method_name: Name of interpolation method (case-insensitive)
        
    Returns:
        OpenCV interpolation constant
        
    Raises:
        ValueError: If method name is not supported
    """
    method_name = method_name.upper()
    try:
        return InterpolationMethod[method_name].value
    except KeyError:
        raise ValueError(
            f"Unknown interpolation method: {method_name}. "
            f"Supported: {', '.join([m.name for m in InterpolationMethod])}"
        )


def upscale_image(
    image: np.ndarray,
    scale_factor: float,
    interpolation: str = "LANCZOS4"
) -> np.ndarray:
    """
    Upscale an image by a given scale factor.
    
    Args:
        image: Input image as numpy array
        scale_factor: Scale factor (e.g., 2.0 for 2x upscaling)
        interpolation: Interpolation method (default: LANCZOS4)
        
    Returns:
        Upscaled image as numpy array
        
    Raises:
        ValueError: If scale factor is invalid or interpolation method unknown
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    if scale_factor <= 0:
        raise ValueError(f"Scale factor must be positive, got {scale_factor}")
    
    if scale_factor == 1.0:
        logger.debug("Scale factor is 1.0, returning original image")
        return image.copy()
    
    # Get interpolation method
    interp = get_interpolation_method(interpolation)
    
    # Calculate new dimensions
    height, width = image.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Upscale
    upscaled = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interp
    )
    
    logger.info(
        f"Image upscaled: {width}x{height} -> {new_width}x{new_height} "
        f"({scale_factor}x, {interpolation})"
    )
    
    return upscaled


def upscale_image_multiple(
    image: np.ndarray,
    scale_factors: list[float] = [2.0, 4.0, 8.0],
    interpolation: str = "LANCZOS4"
) -> dict[float, np.ndarray]:
    """
    Upscale an image to multiple scales.
    
    Args:
        image: Input image as numpy array
        scale_factors: List of scale factors (default: [2.0, 4.0, 8.0])
        interpolation: Interpolation method (default: LANCZOS4)
        
    Returns:
        Dictionary mapping scale factors to upscaled images
    """
    results = {}
    
    for scale in scale_factors:
        try:
            upscaled = upscale_image(image, scale, interpolation)
            results[scale] = upscaled
        except Exception as e:
            logger.error(f"Failed to upscale at {scale}x: {e}")
            raise
    
    logger.info(f"Generated {len(results)} upscaled images")
    return results


def resize_to_size(
    image: np.ndarray,
    width: int,
    height: int,
    interpolation: str = "LANCZOS4",
    keep_aspect_ratio: bool = False
) -> np.ndarray:
    """
    Resize an image to a specific size.
    
    Args:
        image: Input image as numpy array
        width: Target width
        height: Target height
        interpolation: Interpolation method (default: LANCZOS4)
        keep_aspect_ratio: If True, maintains aspect ratio (may not fill target size)
        
    Returns:
        Resized image as numpy array
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    if width <= 0 or height <= 0:
        raise ValueError(f"Target dimensions must be positive: {width}x{height}")
    
    interp = get_interpolation_method(interpolation)
    
    if keep_aspect_ratio:
        # Calculate scale to fit within target size
        src_height, src_width = image.shape[:2]
        scale_w = width / src_width
        scale_h = height / src_height
        scale = min(scale_w, scale_h)
        
        new_width = int(src_width * scale)
        new_height = int(src_height * scale)
    else:
        new_width = width
        new_height = height
    
    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=interp
    )
    
    logger.info(f"Image resized to {new_width}x{new_height} ({interpolation})")
    
    return resized


def compare_interpolation_methods(
    image: np.ndarray,
    scale_factor: float = 2.0
) -> dict[str, np.ndarray]:
    """
    Compare different interpolation methods on the same image.
    Useful for forensic analysis to choose best method.
    
    Args:
        image: Input image as numpy array
        scale_factor: Scale factor to apply (default: 2.0)
        
    Returns:
        Dictionary mapping method names to upscaled images
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    results = {}
    
    for method in InterpolationMethod:
        try:
            upscaled = upscale_image(image, scale_factor, method.name)
            results[method.name] = upscaled
            logger.debug(f"Compared {method.name} interpolation")
        except Exception as e:
            logger.error(f"Failed to compare {method.name}: {e}")
            raise
    
    logger.info(f"Compared {len(results)} interpolation methods")
    return results


def calculate_upscaling_statistics(
    original: np.ndarray,
    upscaled: np.ndarray
) -> dict:
    """
    Calculate statistics about the upscaling operation.
    
    Args:
        original: Original image
        upscaled: Upscaled image
        
    Returns:
        Dictionary containing statistics
    """
    orig_height, orig_width = original.shape[:2]
    upsc_height, upsc_width = upscaled.shape[:2]
    
    scale_h = upsc_height / orig_height
    scale_w = upsc_width / orig_width
    
    pixel_increase = (upsc_height * upsc_width) / (orig_height * orig_width)
    
    stats = {
        "original_dimensions": (orig_width, orig_height),
        "upscaled_dimensions": (upsc_width, upsc_height),
        "scale_factor_width": round(scale_w, 2),
        "scale_factor_height": round(scale_h, 2),
        "pixel_increase_ratio": round(pixel_increase, 2),
        "original_pixels": orig_height * orig_width,
        "upscaled_pixels": upsc_height * upsc_width
    }
    
    return stats
