"""
Deblurring and motion blur removal module.
Implements experimental deblurring methods for forensic analysis.

IMPORTANT: These methods are EXPERIMENTAL and results should NOT be treated as
recovered ground truth. Deblurring algorithms cannot recover information not
present in the original image.
"""

import logging
from typing import Tuple

import cv2
import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


def apply_wiener_filter(
    image: np.ndarray,
    noise_variance: float = 10.0,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Apply Wiener filtering for deblurring.
    
    Wiener filtering attempts to minimize mean square error between the
    estimated and original image. It works well when the noise
    characteristics are known.
    
    WARNING: This is EXPERIMENTAL. Results may not represent actual
    license plate characters and should be manually verified.
    
    Args:
        image: Input image (BGR or grayscale)
        noise_variance: Estimated noise variance (default: 10.0, range typically 5-50)
        kernel_size: Blur kernel size (default: 5)
        
    Returns:
        Deblurred image
        
    Raises:
        ValueError: If image is invalid or parameters are invalid
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    if noise_variance < 0:
        raise ValueError(f"Noise variance must be non-negative, got {noise_variance}")
    
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"Kernel size must be positive odd, got {kernel_size}")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Convert to float
    image_float = gray.astype(np.float64)
    
    # Create motion blur kernel (approximation of common blur)
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    
    # Apply Wiener-like filtering using frequency domain
    # This is a simplified implementation
    f_transform = np.fft.fft2(image_float)
    
    # Power spectrum
    power_spectrum = np.abs(f_transform) ** 2
    
    # Wiener filter: H* / (|H|^2 + noise_variance / signal_power)
    # Simplified version
    wiener_kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + noise_variance + 1e-8)
    
    # Apply in spatial domain using scipy
    deblurred = signal.wiener(gray, mysize=kernel_size)
    
    # Convert back to uint8
    deblurred = np.clip(deblurred, 0, 255).astype(image.dtype)
    
    logger.info(
        f"Wiener filter applied: kernel_size={kernel_size}, "
        f"noise_variance={noise_variance} (EXPERIMENTAL)"
    )
    
    return deblurred


def apply_richardson_lucy_deconvolution(
    image: np.ndarray,
    kernel_size: int = 5,
    iterations: int = 10,
    regularization: float = 0.0
) -> np.ndarray:
    """
    Apply Richardson-Lucy deconvolution for deblurring.
    
    Richardson-Lucy is an iterative algorithm that assumes Poisson noise
    and can recover point spread function effects. It's more sophisticated
    than Wiener filtering but slower.
    
    WARNING: This is EXPERIMENTAL. Results may not represent actual
    license plate characters and should be manually verified.
    
    Args:
        image: Input image (BGR or grayscale)
        kernel_size: Size of blur kernel (default: 5)
        iterations: Number of iterations (default: 10, range 5-30)
        regularization: Regularization parameter (default: 0.0, range 0-1)
        
    Returns:
        Deblurred image
        
    Raises:
        ValueError: If image is invalid or parameters are invalid
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"Kernel size must be positive odd, got {kernel_size}")
    
    if iterations <= 0:
        raise ValueError(f"Iterations must be positive, got {iterations}")
    
    if regularization < 0 or regularization > 1:
        raise ValueError(f"Regularization must be 0-1, got {regularization}")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Convert to float
    image_float = gray.astype(np.float64)
    image_float = image_float / 255.0  # Normalize to 0-1
    
    # Create motion blur kernel
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    
    # Initialize estimate as input image
    estimate = image_float.copy()
    
    # Richardson-Lucy iteration
    for iteration in range(iterations):
        # Convolve estimate with kernel
        convolved = cv2.filter2D(estimate, -1, kernel)
        
        # Avoid division by zero
        convolved = np.maximum(convolved, 1e-10)
        
        # Error term
        error = image_float / convolved
        
        # Convolve error with flipped kernel
        kernel_flipped = cv2.flip(kernel, -1)
        correction = cv2.filter2D(error, -1, kernel_flipped)
        
        # Update estimate
        estimate = estimate * correction
        
        # Apply regularization
        if regularization > 0:
            estimate = (1 - regularization) * estimate + regularization * image_float
        
        # Clip to valid range
        estimate = np.clip(estimate, 0, 1)
    
    # Convert back to uint8
    deblurred = (estimate * 255).astype(image.dtype)
    
    logger.info(
        f"Richardson-Lucy deconvolution applied: kernel_size={kernel_size}, "
        f"iterations={iterations}, regularization={regularization} (EXPERIMENTAL)"
    )
    
    return deblurred


def apply_motion_blur_removal(
    image: np.ndarray,
    kernel_size: int = 15,
    angle: float = 0.0,
    iterations: int = 5
) -> np.ndarray:
    """
    Apply motion blur removal using estimated motion kernel.
    
    This method estimates and removes motion blur by creating a kernel
    that approximates linear motion blur at a given angle.
    
    WARNING: This is EXPERIMENTAL. Results may not represent actual
    license plate characters and should be manually verified.
    
    Args:
        image: Input image (BGR or grayscale)
        kernel_size: Size of motion kernel (default: 15, odd number)
        angle: Direction of motion in degrees (default: 0.0, range 0-180)
        iterations: Number of iterations (default: 5)
        
    Returns:
        Motion blur removed image
        
    Raises:
        ValueError: If image is invalid or parameters are invalid
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"Kernel size must be positive odd, got {kernel_size}")
    
    if angle < 0 or angle > 180:
        raise ValueError(f"Angle must be 0-180, got {angle}")
    
    if iterations <= 0:
        raise ValueError(f"Iterations must be positive, got {iterations}")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Convert to float
    image_float = gray.astype(np.float64)
    image_float = image_float / 255.0
    
    # Create motion blur kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Create motion kernel using rotation
    size = kernel_size
    kernel = np.zeros((size, size), dtype=np.float32)
    
    # Create line kernel based on angle
    x_center = y_center = size // 2
    angle_rad = np.radians(angle)
    
    for i in range(size):
        for j in range(size):
            dx = i - x_center
            dy = j - y_center
            # Project onto motion direction
            projection = dx * np.cos(angle_rad) + dy * np.sin(angle_rad)
            
            if abs(projection) <= size / 4:
                kernel[i, j] = 1
    
    # Normalize kernel
    kernel = kernel / (kernel.sum() + 1e-8)
    
    # Apply iterative motion blur removal
    estimate = image_float.copy()
    
    for iteration in range(iterations):
        # Convolve with motion kernel
        convolved = cv2.filter2D(estimate, -1, kernel)
        convolved = np.clip(convolved, 1e-10, 1)
        
        # Inverse filter
        inverse_est = image_float / convolved
        inverse_est = np.clip(inverse_est, 0, 1)
        
        # Update estimate
        estimate = 0.5 * estimate + 0.5 * inverse_est
    
    # Convert back to uint8
    deblurred = (estimate * 255).astype(image.dtype)
    
    logger.info(
        f"Motion blur removal applied: kernel_size={kernel_size}, "
        f"angle={angle:.1f}°, iterations={iterations} (EXPERIMENTAL)"
    )
    
    return deblurred


def apply_laplacian_sharpening_deblur(
    image: np.ndarray,
    strength: float = 1.0
) -> np.ndarray:
    """
    Apply Laplacian-based edge enhancement for deblurring effect.
    
    This is a simpler deblurring approach that enhances edges but doesn't
    actually reverse blur. It's faster but less effective than true
    deconvolution methods.
    
    Args:
        image: Input image (BGR or grayscale)
        strength: Strength of sharpening (default: 1.0)
        
    Returns:
        Edge-enhanced image
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    image_float = gray.astype(np.float32)
    
    # Calculate Laplacian
    laplacian = cv2.Laplacian(image_float, cv2.CV_32F)
    
    # Apply sharpening
    sharpened = image_float - strength * laplacian
    sharpened = np.clip(sharpened, 0, 255).astype(image.dtype)
    
    logger.info(f"Laplacian edge enhancement applied: strength={strength}")
    
    return sharpened


def calculate_blur_metrics(image: np.ndarray) -> dict:
    """
    Calculate metrics to estimate blur level in image.
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        Dictionary with blur metrics
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image array")
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian variance (inverse of blur: higher = sharper)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = laplacian.var()
    
    # Calculate Sobel edge magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2).mean()
    
    # Calculate FFT spectrum to detect blur
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # High frequency content (inverse of blur)
    center_size = min(magnitude_spectrum.shape) // 4
    cy, cx = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
    center_region = magnitude_spectrum[
        cy - center_size:cy + center_size,
        cx - center_size:cx + center_size
    ]
    high_freq = magnitude_spectrum.sum() - center_region.sum()
    
    metrics = {
        "laplacian_variance": float(laplacian_var),
        "edge_magnitude": float(edge_magnitude),
        "high_frequency_content": float(high_freq),
        "is_blurry": laplacian_var < 100,
        "blur_severity": max(0, min(100, 100 * (1 - laplacian_var / 500)))  # 0-100 scale
    }
    
    logger.debug(f"Blur metrics: {metrics}")
    
    return metrics
