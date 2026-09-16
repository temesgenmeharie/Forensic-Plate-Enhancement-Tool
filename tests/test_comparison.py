"""Tests for comparison and visualization module."""

import unittest

import cv2
import numpy as np

from app.comparison.comparison import ComparisonGrid, AnnotatedComparison, ComparisonReport


class TestComparisonGrid(unittest.TestCase):
    """Test comparison grid creation."""
    
    def setUp(self):
        """Create test images."""
        self.img1 = np.ones((480, 640, 3), dtype=np.uint8) * 100
        self.img2 = np.ones((480, 640, 3), dtype=np.uint8) * 150
        self.img3 = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    def test_create_comparison_grid_two_images(self):
        """Test creating grid with two images."""
        images = {"image1": self.img1, "image2": self.img2}
        
        grid = ComparisonGrid.create_comparison_grid(images)
        
        self.assertIsNotNone(grid)
        self.assertEqual(len(grid.shape), 3)  # Should be color image
        self.assertGreater(grid.shape[0], self.img1.shape[0])  # Has label height
    
    def test_create_comparison_grid_with_labels(self):
        """Test grid creation with custom labels."""
        images = {"image1": self.img1, "image2": self.img2}
        labels = {"image1": "ORIGINAL", "image2": "ENHANCED"}
        
        grid = ComparisonGrid.create_comparison_grid(images, labels)
        
        self.assertIsNotNone(grid)
    
    def test_create_comparison_grid_four_images(self):
        """Test grid creation with four images (2x2)."""
        images = {
            "image1": self.img1,
            "image2": self.img2,
            "image3": self.img3,
            "image4": self.img1.copy()
        }
        
        grid = ComparisonGrid.create_comparison_grid(images, grid_cols=2)
        
        self.assertIsNotNone(grid)
        # Grid should be approximately 2x the width and height
        self.assertGreater(grid.shape[0], self.img1.shape[0] * 1.5)
        self.assertGreater(grid.shape[1], self.img1.shape[1] * 1.5)
    
    def test_before_after_comparison(self):
        """Test before-after comparison."""
        comparison = ComparisonGrid.create_before_after_comparison(
            self.img1,
            self.img2,
            before_label="ORIGINAL",
            after_label="ENHANCED"
        )
        
        self.assertIsNotNone(comparison)
        self.assertEqual(len(comparison.shape), 3)
    
    def test_processing_stages_grid(self):
        """Test processing stages grid."""
        stages = {
            "Original": self.img1,
            "Stage1": self.img2,
            "Stage2": self.img3
        }
        
        grid = ComparisonGrid.create_processing_stages_grid(stages)
        
        self.assertIsNotNone(grid)
    
    def test_comparison_grid_no_images(self):
        """Test that empty images dict raises error."""
        with self.assertRaises(ValueError):
            ComparisonGrid.create_comparison_grid({})
    
    def test_comparison_grid_grayscale(self):
        """Test grid with grayscale images."""
        gray1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        
        images = {"gray1": gray1, "gray2": gray2}
        
        grid = ComparisonGrid.create_comparison_grid(images)
        
        self.assertIsNotNone(grid)


class TestAnnotatedComparison(unittest.TestCase):
    """Test annotated comparison functionality."""
    
    def setUp(self):
        """Create test images."""
        self.img1 = np.ones((480, 640, 3), dtype=np.uint8) * 100
        self.img2 = self.img1.copy()
        
        # Add some difference to img2
        self.img2[100:200, 150:250] = [200, 200, 200]
    
    def test_draw_roi_boxes(self):
        """Test drawing ROI boxes."""
        rois = [(100, 100, 100, 100), (300, 200, 150, 100)]
        
        annotated = AnnotatedComparison.draw_roi_boxes(self.img1, rois)
        
        self.assertEqual(annotated.shape, self.img1.shape)
        # Check that image was modified
        self.assertFalse(np.array_equal(annotated, self.img1))
    
    def test_draw_roi_boxes_empty(self):
        """Test drawing with no ROIs."""
        rois = []
        
        annotated = AnnotatedComparison.draw_roi_boxes(self.img1, rois)
        
        self.assertTrue(np.array_equal(annotated, self.img1))
    
    def test_draw_difference_highlight(self):
        """Test highlighting differences."""
        highlighted = AnnotatedComparison.draw_difference_highlight(
            self.img1,
            self.img2,
            threshold=30
        )
        
        self.assertEqual(highlighted.shape, self.img1.shape)
    
    def test_add_text_annotation(self):
        """Test adding text annotation."""
        text = "TEST ANNOTATION"
        
        annotated = AnnotatedComparison.add_text_annotation(
            self.img1,
            text,
            position=(10, 30)
        )
        
        self.assertEqual(annotated.shape, self.img1.shape)
        # Image should be modified
        self.assertFalse(np.array_equal(annotated, self.img1))
    
    def test_add_text_annotation_with_background(self):
        """Test text annotation with background."""
        text = "TEST ANNOTATION"
        
        annotated = AnnotatedComparison.add_text_annotation(
            self.img1,
            text,
            position=(10, 30),
            bg_color=(0, 0, 255)
        )
        
        self.assertEqual(annotated.shape, self.img1.shape)


class TestComparisonReport(unittest.TestCase):
    """Test comparison report generation."""
    
    def setUp(self):
        """Create test images."""
        self.original = np.ones((480, 640, 3), dtype=np.uint8) * 100
        self.enhanced1 = np.ones((480, 640, 3), dtype=np.uint8) * 120
        self.enhanced2 = np.ones((480, 640, 3), dtype=np.uint8) * 140
    
    def test_create_forensic_comparison_sheet(self):
        """Test creating forensic comparison sheet."""
        enhancements = {
            "Enhancement1": self.enhanced1,
            "Enhancement2": self.enhanced2
        }
        
        sheet = ComparisonReport.create_forensic_comparison_sheet(
            self.original,
            enhancements,
            title="TEST FORENSIC ANALYSIS"
        )
        
        self.assertIsNotNone(sheet)
        self.assertEqual(len(sheet.shape), 3)
        # Should have title area
        self.assertGreater(sheet.shape[0], self.original.shape[0])
    
    def test_create_forensic_comparison_sheet_with_labels(self):
        """Test forensic sheet with custom labels."""
        enhancements = {"enhancement": self.enhanced1}
        labels = {"enhancement": "CLAHE ENHANCED"}
        
        sheet = ComparisonReport.create_forensic_comparison_sheet(
            self.original,
            enhancements,
            enhancement_labels=labels
        )
        
        self.assertIsNotNone(sheet)
    
    def test_create_evidence_chain_visualization(self):
        """Test evidence chain visualization."""
        processed = {
            "Step1": self.enhanced1,
            "Step2": self.enhanced2
        }
        
        chain = ComparisonReport.create_evidence_chain_visualization(
            "EV-TEST-001",
            self.original,
            processed
        )
        
        self.assertIsNotNone(chain)
        self.assertEqual(len(chain.shape), 3)
        # Should include header
        self.assertGreater(chain.shape[0], self.original.shape[0])
    
    def test_create_evidence_chain_with_hashes(self):
        """Test evidence chain with hash information."""
        processed = {"enhanced": self.enhanced1}
        hashes = {
            "original": "abc123",
            "enhanced": "def456"
        }
        
        chain = ComparisonReport.create_evidence_chain_visualization(
            "EV-TEST-002",
            self.original,
            processed,
            hashes=hashes
        )
        
        self.assertIsNotNone(chain)


if __name__ == "__main__":
    unittest.main()
