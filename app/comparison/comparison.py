"""
Comparison interface for side-by-side forensic analysis.
Creates labeled comparison grids and visual analysis tools.
"""

import logging
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ComparisonGrid:
    """Create side-by-side comparison grids of multiple images."""
    
    @staticmethod
    def create_comparison_grid(
        images: Dict[str, np.ndarray],
        labels: Optional[Dict[str, str]] = None,
        grid_cols: int = 2,
        label_height: int = 40,
        font_scale: float = 1.0,
        font_color: Tuple[int, int, int] = (255, 255, 255),
        bg_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        Create a comparison grid with labeled images.
        
        Args:
            images: Dictionary of image_name -> image_array
            labels: Optional dictionary of image_name -> label_text
            grid_cols: Number of columns in grid (default: 2)
            label_height: Height of label area (default: 40)
            font_scale: Font scale for labels (default: 1.0)
            font_color: Label font color in BGR (default: white)
            bg_color: Label background color in BGR (default: black)
            
        Returns:
            Comparison grid image
        """
        if not images:
            raise ValueError("No images provided")
        
        labels = labels or {}
        image_list = list(images.items())
        num_images = len(image_list)
        grid_rows = (num_images + grid_cols - 1) // grid_cols
        
        # Get image dimensions (assume all same size)
        first_img = image_list[0][1]
        img_height, img_width = first_img.shape[:2]
        
        # Create grid canvas
        total_width = grid_cols * img_width
        total_height = grid_rows * (img_height + label_height)
        
        # Handle different image types
        if len(first_img.shape) == 3:
            grid = np.zeros((total_height, total_width, 3), dtype=first_img.dtype)
        else:
            grid = np.zeros((total_height, total_width), dtype=first_img.dtype)
        
        # Place images and labels
        for idx, (img_name, img) in enumerate(image_list):
            row = idx // grid_cols
            col = idx % grid_cols
            
            y_start = row * (img_height + label_height)
            x_start = col * img_width
            
            # Resize image if needed
            if img.shape != first_img.shape:
                img = cv2.resize(img, (img_width, img_height))
            
            # Place image
            grid[y_start:y_start + img_height, x_start:x_start + img_width] = img
            
            # Add label
            label = labels.get(img_name, img_name)
            label_y = y_start + img_height + label_height - 5
            
            # Add background for label
            cv2.rectangle(
                grid,
                (x_start, y_start + img_height),
                (x_start + img_width, y_start + img_height + label_height),
                bg_color,
                -1
            )
            
            # Add text
            cv2.putText(
                grid,
                label,
                (x_start + 5, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_color,
                1
            )
        
        logger.info(f"Comparison grid created: {grid_rows}x{grid_cols}")
        
        return grid
    
    @staticmethod
    def create_before_after_comparison(
        before_image: np.ndarray,
        after_image: np.ndarray,
        before_label: str = "ORIGINAL",
        after_label: str = "ENHANCED",
        label_height: int = 40
    ) -> np.ndarray:
        """
        Create a before-after comparison image.
        
        Args:
            before_image: Original/before image
            after_image: Processed/after image
            before_label: Label for before image (default: "ORIGINAL")
            after_label: Label for after image (default: "ENHANCED")
            label_height: Height of label area (default: 40)
            
        Returns:
            Side-by-side comparison image
        """
        images = {
            "before": before_image,
            "after": after_image
        }
        
        labels = {
            "before": before_label,
            "after": after_label
        }
        
        return ComparisonGrid.create_comparison_grid(
            images,
            labels,
            grid_cols=2,
            label_height=label_height
        )
    
    @staticmethod
    def create_processing_stages_grid(
        stages: Dict[str, np.ndarray],
        stage_labels: Optional[Dict[str, str]] = None
    ) -> np.ndarray:
        """
        Create grid showing multiple processing stages.
        
        Args:
            stages: Dictionary of stage_name -> image
            stage_labels: Optional custom labels for each stage
            
        Returns:
            Processing stages comparison grid
        """
        if not stages:
            raise ValueError("No stages provided")
        
        # Determine optimal grid layout
        num_stages = len(stages)
        
        if num_stages <= 2:
            grid_cols = 2
        elif num_stages <= 4:
            grid_cols = 2
        elif num_stages <= 6:
            grid_cols = 3
        else:
            grid_cols = 4
        
        stage_labels = stage_labels or {name: name for name in stages.keys()}
        
        return ComparisonGrid.create_comparison_grid(
            stages,
            stage_labels,
            grid_cols=grid_cols,
            label_height=50,
            font_scale=0.8
        )


class AnnotatedComparison:
    """Create annotated comparisons with highlighting and arrows."""
    
    @staticmethod
    def draw_roi_boxes(
        image: np.ndarray,
        rois: List[Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw regions of interest (bounding boxes) on image.
        
        Args:
            image: Input image
            rois: List of (x, y, width, height) regions
            color: Box color in BGR (default: green)
            thickness: Box line thickness (default: 2)
            
        Returns:
            Image with drawn boxes
        """
        annotated = image.copy()
        
        for x, y, w, h in rois:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        
        logger.debug(f"Drew {len(rois)} ROI boxes")
        
        return annotated
    
    @staticmethod
    def draw_difference_highlight(
        image1: np.ndarray,
        image2: np.ndarray,
        threshold: int = 30,
        highlight_color: Tuple[int, int, int] = (0, 0, 255)
    ) -> np.ndarray:
        """
        Highlight differences between two images.
        
        Args:
            image1: First image
            image2: Second image
            threshold: Difference threshold (default: 30)
            highlight_color: Color for highlights in BGR (default: red)
            
        Returns:
            Image with highlighted differences
        """
        if image1.shape != image2.shape:
            logger.warning("Images have different shapes")
            return image1.copy()
        
        # Convert to grayscale for comparison
        if len(image1.shape) == 3:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = image1
            gray2 = image2
        
        # Compute difference
        diff = cv2.absdiff(gray1, gray2)
        _, diff_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Create output image
        annotated = image1.copy()
        
        # Apply highlight color where differences exist
        if len(image1.shape) == 3:
            annotated[diff_mask > 0] = highlight_color
        else:
            # For grayscale, increase intensity
            annotated[diff_mask > 0] = 255
        
        logger.debug("Differences highlighted")
        
        return annotated
    
    @staticmethod
    def add_text_annotation(
        image: np.ndarray,
        text: str,
        position: Tuple[int, int] = (10, 30),
        font_scale: float = 1.0,
        color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 2,
        bg_color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Add text annotation to image.
        
        Args:
            image: Input image
            text: Text to add
            position: (x, y) position of text (default: (10, 30))
            font_scale: Font scale (default: 1.0)
            color: Text color in BGR (default: white)
            thickness: Text thickness (default: 2)
            bg_color: Background color (optional)
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Add background if specified
        if bg_color is not None:
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            x, y = position
            cv2.rectangle(
                annotated,
                (x - 5, y - text_size[1] - 5),
                (x + text_size[0] + 5, y + 5),
                bg_color,
                -1
            )
        
        cv2.putText(
            annotated,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness
        )
        
        logger.debug(f"Text annotation added: {text}")
        
        return annotated


class ComparisonReport:
    """Generate comparison reports for forensic analysis."""
    
    @staticmethod
    def create_forensic_comparison_sheet(
        original: np.ndarray,
        enhancements: Dict[str, np.ndarray],
        enhancement_labels: Optional[Dict[str, str]] = None,
        title: str = "FORENSIC ANALYSIS COMPARISON",
        include_metrics: bool = True
    ) -> np.ndarray:
        """
        Create a forensic comparison sheet with original and multiple enhancements.
        
        Args:
            original: Original image
            enhancements: Dictionary of enhancement_name -> enhanced_image
            enhancement_labels: Custom labels for enhancements
            title: Sheet title
            include_metrics: Whether to include quality metrics (default: True)
            
        Returns:
            Comparison sheet image
        """
        # Prepare all images
        all_images = {"ORIGINAL": original}
        all_images.update(enhancements)
        
        # Prepare labels
        labels = enhancement_labels or {}
        labels["ORIGINAL"] = "ORIGINAL EVIDENCE"
        
        # Create comparison grid
        comparison = ComparisonGrid.create_comparison_grid(
            all_images,
            labels,
            grid_cols=min(3, len(all_images)),
            label_height=60,
            font_scale=1.2
        )
        
        # Add title
        title_height = 80
        if len(comparison.shape) == 3:
            title_img = np.zeros((title_height, comparison.shape[1], 3), dtype=comparison.dtype)
        else:
            title_img = np.zeros((title_height, comparison.shape[1]), dtype=comparison.dtype)
        
        # Add title text
        cv2.putText(
            title_img,
            title,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            2
        )
        
        # Combine title and comparison
        result = np.vstack([title_img, comparison])
        
        logger.info("Forensic comparison sheet created")
        
        return result
    
    @staticmethod
    def create_evidence_chain_visualization(
        evidence_id: str,
        original_image: np.ndarray,
        processed_images: Dict[str, np.ndarray],
        hashes: Optional[Dict[str, str]] = None
    ) -> np.ndarray:
        """
        Create visualization showing evidence chain from original to processed.
        
        Args:
            evidence_id: Evidence identifier
            original_image: Original evidence image
            processed_images: Dictionary of process_name -> processed_image
            hashes: Optional dictionary of image_name -> sha256_hash
            
        Returns:
            Evidence chain visualization
        """
        # Create processing stages dict
        stages = {"ORIGINAL": original_image}
        stages.update(processed_images)
        
        # Create comparison grid
        comparison = ComparisonGrid.create_processing_stages_grid(stages)
        
        # Add evidence ID header
        header_height = 60
        if len(comparison.shape) == 3:
            header = np.zeros((header_height, comparison.shape[1], 3), dtype=comparison.dtype)
        else:
            header = np.zeros((header_height, comparison.shape[1]), dtype=comparison.dtype)
        
        header_text = f"EVIDENCE CHAIN: {evidence_id}"
        cv2.putText(
            header,
            header_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2
        )
        
        result = np.vstack([header, comparison])
        
        logger.info(f"Evidence chain visualization created for {evidence_id}")
        
        return result
