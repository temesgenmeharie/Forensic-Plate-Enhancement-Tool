"""
Advanced deblurring specifically designed for license plate recovery.
Focuses on recovering invisible/blurred plate numbers.

⚠️ EXPERIMENTAL: Results do NOT recover ground truth. Manual verification required.
"""

import logging
from typing import Optional, Tuple
import cv2
import numpy as np
from scipy import signal, ndimage

logger = logging.getLogger(__name__)


def apply_blind_deconvolution(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    iterations: int = 50,
    regularization: float = 0.01
) -> np.ndarray:
    """
    Blind deconvolution - estimate both image and PSF (Point Spread Function).
    
    Useful for plates with unknown blur type.
    
    Args:
        image: Input blurred image
        kernel_size: Estimated PSF kernel size
        iterations: Number of iterations (default: 50)
        regularization: Regularization strength (default: 0.01)
        
    Returns:
        Deblurred image
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Initialize PSF kernel (Gaussian)
    k_h, k_w = kernel_size
    psf = cv2.getGaussianKernel(k_h, 1.5)
    psf = psf @ cv2.getGaussianKernel(k_w, 1.5).T
    psf = psf / psf.sum()
    
    # Normalize
    gray = gray.astype(np.float64) / 255.0
    
    # Iterative blind deconvolution
    for _ in range(iterations):
        # Estimate image (Wiener-like filtering)
        denominator = signal.convolve2d(
            psf,
            psf[::-1, ::-1],
            mode='same'
        ) + regularization
        
        numerator = signal.convolve2d(gray, psf[::-1, ::-1], mode='same')
        
        gray = numerator / denominator
        gray = np.clip(gray, 0, 1)
    
    # Convert back to uint8
    result = np.clip(gray * 255, 0, 255).astype(np.uint8)
    
    logger.info(f"Blind deconvolution applied: kernel={kernel_size}, iterations={iterations}")
    
    return result


def apply_lucy_richardson_advanced(
    image: np.ndarray,
    kernel_size: int = 7,
    iterations: int = 20,
    psf_type: str = "motion"
) -> np.ndarray:
    """
    Advanced Richardson-Lucy with different PSF types.
    
    Args:
        image: Input image
        kernel_size: PSF kernel size
        iterations: Number of iterations
        psf_type: "motion", "gaussian", "uniform"
        
    Returns:
        Deblurred image
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float32) / 255.0
    
    # Create PSF based on type
    if psf_type == "motion":
        # Motion blur (horizontal)
        psf = np.zeros((kernel_size, kernel_size))
        psf[kernel_size // 2, :] = 1
        psf = psf / psf.sum()
    elif psf_type == "gaussian":
        psf = cv2.getGaussianKernel(kernel_size, kernel_size / 2)
        psf = psf @ psf.T
        psf = psf / psf.sum()
    else:  # uniform
        psf = np.ones((kernel_size, kernel_size))
        psf = psf / psf.sum()
    
    # Richardson-Lucy deconvolution
    result = gray.copy()
    psf_mirror = psf[::-1, ::-1]
    
    for i in range(iterations):
        convolved = signal.convolve2d(result, psf, mode='same')
        convolved = np.maximum(convolved, 1e-5)  # Avoid division by zero
        
        correction = signal.convolve2d(gray / convolved, psf_mirror, mode='same')
        result = result * correction
        result = np.clip(result, 0, 1)
    
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    logger.info(f"Advanced Richardson-Lucy: psf_type={psf_type}, iterations={iterations}")
    
    return result


def apply_total_variation_deblur(
    image: np.ndarray,
    strength: float = 0.1,
    iterations: int = 100
) -> np.ndarray:
    """
    Total Variation (TV) denoising for deblurring.
    
    Preserves edges while removing blur artifacts.
    
    Args:
        image: Input image
        strength: Deblurring strength (0.01-0.5)
        iterations: Number of iterations
        
    Returns:
        Deblurred image
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float32) / 255.0
    
    result = gray.copy()
    
    for _ in range(iterations):
        # Compute gradients
        gy, gx = np.gradient(result)
        
        # Compute gradient magnitude
        mag = np.sqrt(gx**2 + gy**2) + 1e-8
        
        # TV update
        div_x = np.roll(gx / mag, 1, axis=1) - gx / mag
        div_y = np.roll(gy / mag, 1, axis=0) - gy / mag
        
        result = result + strength * (div_x + div_y)
        result = np.clip(result, 0, 1)
    
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    logger.info(f"Total Variation deblur: strength={strength}, iterations={iterations}")
    
    return result


def apply_super_resolution_upscale(
    image: np.ndarray,
    scale_factor: int = 2,
    iterations: int = 10
) -> np.ndarray:
    """
    Super-resolution upscaling with iterative refinement.
    
    Combines upscaling with deblurring for better plate visibility.
    
    Args:
        image: Input image
        scale_factor: Upscaling factor (2, 3, or 4)
        iterations: Refinement iterations
        
    Returns:
        Super-resolved image
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image")
    
    if scale_factor not in [2, 3, 4]:
        raise ValueError(f"Scale factor must be 2, 3, or 4, got {scale_factor}")
    
    # Initial upscaling
    h, w = image.shape[:2]
    new_h, new_w = h * scale_factor, w * scale_factor
    upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Iterative refinement with edge enhancement
    result = upscaled.astype(np.float32)
    
    for _ in range(iterations):
        # Apply unsharp masking for edge enhancement
        if len(result.shape) == 3:
            blurred = cv2.GaussianBlur(result, (3, 3), 0.5)
        else:
            blurred = cv2.GaussianBlur(result, (3, 3), 0.5)
        
        result = result + 0.5 * (result - blurred)
        result = np.clip(result, 0, 255)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    logger.info(f"Super-resolution upscale: scale={scale_factor}, iterations={iterations}")
    
    return result


def apply_morphological_enhancement(
    image: np.ndarray,
    kernel_size: int = 5,
    operations: list = None
) -> np.ndarray:
    """
    Morphological operations to enhance plate visibility.
    
    Useful for improving text clarity on blurred plates.
    
    Args:
        image: Input image
        kernel_size: Morphological kernel size
        operations: List of operations ("open", "close", "gradient", "tophat", "blackhat")
        
    Returns:
        Enhanced image
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image")
    
    if operations is None:
        operations = ["close", "open"]
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    result = gray.copy()
    
    for op in operations:
        if op == "open":
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        elif op == "close":
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        elif op == "gradient":
            result = cv2.morphologyEx(result, cv2.MORPH_GRADIENT, kernel)
        elif op == "tophat":
            result = cv2.morphologyEx(result, cv2.MORPH_TOPHAT, kernel)
        elif op == "blackhat":
            result = cv2.morphologyEx(result, cv2.MORPH_BLACKHAT, kernel)
    
    logger.info(f"Morphological enhancement: operations={operations}")
    
    return result


def apply_frequency_domain_deblur(
    image: np.ndarray,
    blur_type: str = "motion",
    strength: float = 1.0
) -> np.ndarray:
    """
    Frequency domain deblurring (Fourier-based).
    
    Works in frequency domain for different blur types.
    
    Args:
        image: Input image
        blur_type: "motion", "gaussian", "uniform"
        strength: Deblurring strength (0.5-2.0)
        
    Returns:
        Deblurred image
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    gray = gray.astype(np.float32)
    
    # FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    
    h, w = gray.shape
    
    # Create inverse filter
    if blur_type == "motion":
        # Motion blur filter (horizontal)
        kernel = np.zeros((h, w), dtype=np.complex64)
        for i in range(w):
            kernel[h//2, i] = 1 / w
    elif blur_type == "gaussian":
        # Gaussian blur filter
        y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
        kernel = np.exp(-(x**2 + y**2) / (2 * (w/10)**2))
    else:  # uniform
        kernel = np.ones((h, w)) / (h * w)
    
    kernel_fft = np.fft.fft2(kernel, s=(h, w))
    kernel_shift = np.fft.fftshift(kernel_fft)
    
    # Wiener filter in frequency domain
    denominator = np.abs(kernel_shift)**2 + 1e-3
    inverse_filter = np.conj(kernel_shift) / denominator
    
    # Apply filter with strength
    f_deblur = f_shift * (1 + strength * inverse_filter)
    
    # IFFT
    f_shift_back = np.fft.ifftshift(f_deblur)
    result = np.fft.ifft2(f_shift_back)
    result = np.abs(result)
    
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    logger.info(f"Frequency domain deblur: blur_type={blur_type}, strength={strength}")
    
    return result


def apply_multi_scale_deblur(
    image: np.ndarray,
    scales: list = None
) -> np.ndarray:
    """
    Multi-scale deblurring (pyramid-based).
    
    Processes at multiple scales for better results.
    
    Args:
        image: Input image
        scales: Scale factors [0.5, 1.0, 2.0] etc
        
    Returns:
        Deblurred image
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image")
    
    if scales is None:
        scales = [0.5, 1.0]
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    results = []
    
    for scale in scales:
        h, w = gray.shape
        new_h, new_w = int(h * scale), int(w * scale)
        
        scaled = cv2.resize(gray, (new_w, new_h))
        
        # Apply Wiener filter
        kernel = np.ones((5, 5)) / 25
        deblurred = cv2.filter2D(scaled, -1, kernel)
        
        # Resize back
        result = cv2.resize(deblurred, (w, h))
        results.append(result)
    
    # Combine results (average)
    final = np.mean(results, axis=0)
    final = np.clip(final, 0, 255).astype(np.uint8)
    
    logger.info(f"Multi-scale deblur: scales={scales}")
    
    return final


def calculate_plate_visibility_score(image: np.ndarray) -> dict:
    """
    Calculate metrics for plate visibility/readability.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with visibility metrics
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image")
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # Contrast (standard deviation)
    contrast = float(np.std(gray))
    
    # Edge density
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_density = np.mean(np.sqrt(sobelx**2 + sobely**2))
    
    # Character definition (texture analysis)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_peaks = np.sum(hist > np.mean(hist))
    
    # Composite score (0-100)
    score = (sharpness / 500.0 + contrast / 100.0 + edge_density / 10.0) * 25
    score = np.clip(score, 0, 100)
    
    metrics = {
        "visibility_score": float(score),
        "sharpness": float(sharpness),
        "contrast": float(contrast),
        "edge_density": float(edge_density),
        "texture_definition": float(hist_peaks),
        "is_readable": score > 30
    }
    
    logger.debug(f"Plate visibility: {metrics}")
    
    return metrics


def recommend_deblur_method(image: np.ndarray) -> Tuple[str, dict]:
    """
    Recommend best deblurring method based on image analysis.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (recommended_method, confidence_scores)
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image")
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Analyze blur characteristics
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(np.abs(fft))
    
    # Motion blur signature (directional)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    motion_score = abs(np.mean(sobelx) - np.mean(sobely)) / (np.mean(sobelx) + np.mean(sobely) + 1e-5)
    
    # Gaussian blur signature (smooth degradation)
    gaussian_score = 1.0 - (laplacian.var() / 1000.0)
    
    # Uniform blur (simple degradation)
    uniform_score = 0.5
    
    # Normalize
    total = motion_score + gaussian_score + uniform_score
    scores = {
        "motion": motion_score / total,
        "gaussian": gaussian_score / total,
        "uniform": uniform_score / total,
        "blind": 0.3  # Always include blind deconvolution
    }
    
    # Recommend best method
    best_method = max(scores, key=scores.get)
    
    logger.info(f"Recommended deblur method: {best_method} (scores: {scores})")
    
    return best_method, scores
