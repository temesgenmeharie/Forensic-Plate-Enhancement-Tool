"""
GUI utility functions for image selection and interaction.
Provides interactive image viewing and region selection.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RegionSelector:
    """
    Interactive region selector using OpenCV mouse events.
    Allows user to select a rectangular region on an image.
    """
    
    def __init__(self, image: np.ndarray, window_name: str = "Select Region"):
        """
        Initialize region selector.
        
        Args:
            image: Image to display
            window_name: Name of OpenCV window
        """
        self.image = image.copy()
        self.display_image = image.copy()
        self.window_name = window_name
        
        self.start_point: Optional[Tuple[int, int]] = None
        self.end_point: Optional[Tuple[int, int]] = None
        self.is_selecting = False
        self.completed = False
        
        # Scale image if too large for screen
        self.scale = 1.0
        screen_width, screen_height = 1920, 1080  # Approximate screen size
        img_height, img_width = image.shape[:2]
        
        if img_width > screen_width or img_height > screen_height:
            scale_w = screen_width / img_width
            scale_h = screen_height / img_height
            self.scale = min(scale_w, scale_h)
            
            new_width = int(img_width * self.scale)
            new_height = int(img_height * self.scale)
            self.display_image = cv2.resize(image, (new_width, new_height))
        
        logger.debug(f"Region selector initialized (scale: {self.scale})")
    
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.is_selecting = True
            logger.debug(f"Selection started at {self.start_point}")
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_selecting and self.start_point:
                self.end_point = (x, y)
                # Draw preview rectangle
                preview = self.display_image.copy()
                cv2.rectangle(preview, self.start_point, self.end_point, (0, 255, 0), 2)
                cv2.imshow(self.window_name, preview)
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.start_point:
                self.end_point = (x, y)
                self.is_selecting = False
                self.completed = True
                
                # Draw final rectangle
                preview = self.display_image.copy()
                cv2.rectangle(preview, self.start_point, self.end_point, (0, 255, 0), 2)
                cv2.imshow(self.window_name, preview)
                
                logger.debug(f"Selection completed: {self.start_point} to {self.end_point}")
    
    def select_region(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Display image and allow user to select region.
        
        Returns:
            Tuple of (x, y, width, height) if region selected, None if cancelled
        """
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        cv2.imshow(self.window_name, self.display_image)
        
        print("Select region by clicking and dragging")
        print("Press ENTER to confirm or ESC to cancel")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                logger.info("Region selection cancelled")
                cv2.destroyWindow(self.window_name)
                return None
            
            elif key == 13:  # ENTER
                if self.completed and self.start_point and self.end_point:
                    # Scale back to original coordinates
                    x1 = int(self.start_point[0] / self.scale)
                    y1 = int(self.start_point[1] / self.scale)
                    x2 = int(self.end_point[0] / self.scale)
                    y2 = int(self.end_point[1] / self.scale)
                    
                    # Normalize coordinates
                    x = min(x1, x2)
                    y = min(y1, y2)
                    width = abs(x2 - x1)
                    height = abs(y2 - y1)
                    
                    logger.info(f"Region selected: ({x}, {y}, {width}, {height})")
                    cv2.destroyWindow(self.window_name)
                    
                    return (x, y, width, height)
                else:
                    print("Please select a region first")
        
        cv2.destroyWindow(self.window_name)
        return None


def display_image(image: np.ndarray, title: str = "Image", wait_key: bool = True) -> None:
    """
    Display an image in a window.
    
    Args:
        image: Image as numpy array
        title: Window title
        wait_key: If True, wait for key press to close
    """
    # Scale image if needed
    height, width = image.shape[:2]
    if width > 1920 or height > 1080:
        scale = min(1920 / width, 1080 / height)
        display = cv2.resize(image, (int(width * scale), int(height * scale)))
    else:
        display = image
    
    cv2.imshow(title, display)
    
    if wait_key:
        print(f"Displaying '{title}' - Press any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def create_comparison_grid(
    images: dict[str, np.ndarray],
    titles: Optional[list[str]] = None,
    grid_cols: int = 2
) -> np.ndarray:
    """
    Create a grid of images for comparison.
    
    Args:
        images: Dictionary of image name -> image array
        titles: Optional list of titles (uses keys if not provided)
        grid_cols: Number of columns in grid (default: 2)
        
    Returns:
        Composite image array
    """
    if not images:
        raise ValueError("No images provided")
    
    # Get image dimensions
    first_image = next(iter(images.values()))
    img_height, img_width = first_image.shape[:2]
    
    # Calculate grid dimensions
    num_images = len(images)
    grid_rows = (num_images + grid_cols - 1) // grid_cols
    
    # Create blank grid
    grid_height = grid_rows * img_height
    grid_width = grid_cols * img_width
    
    # Handle different image types
    if len(first_image.shape) == 3:
        grid = np.zeros((grid_height, grid_width, 3), dtype=first_image.dtype)
    else:
        grid = np.zeros((grid_height, grid_width), dtype=first_image.dtype)
    
    # Place images in grid
    titles = titles or list(images.keys())
    
    for idx, (image_key, image) in enumerate(images.items()):
        row = idx // grid_cols
        col = idx % grid_cols
        
        y_start = row * img_height
        x_start = col * img_width
        
        # Resize if needed
        if image.shape != first_image.shape:
            image = cv2.resize(image, (img_width, img_height))
        
        grid[y_start:y_start + img_height, x_start:x_start + img_width] = image
    
    logger.info(f"Created comparison grid: {grid_rows}x{grid_cols}")
    
    return grid
