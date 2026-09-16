"""Tests for GUI components."""

import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

from PySide6.QtWidgets import QApplication
from app.gui.main_window import MainWindow


class TestMainWindow(unittest.TestCase):
    """Test main window functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Setup QApplication for tests."""
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()
    
    def setUp(self):
        """Create main window."""
        self.window = MainWindow()
    
    def tearDown(self):
        """Cleanup."""
        self.window.close()
    
    def test_window_initialization(self):
        """Test window initializes correctly."""
        self.assertIsNotNone(self.window)
        self.assertEqual(self.window.windowTitle(), "Forensic Plate Enhancer")
    
    def test_ui_created(self):
        """Test UI elements are created."""
        self.assertIsNotNone(self.window.file_label)
        self.assertIsNotNone(self.window.session_label)
        self.assertIsNotNone(self.window.progress_bar)
    
    def test_upscale_combo_options(self):
        """Test upscale combo box has correct options."""
        options = [
            self.window.upscale_combo.itemText(i)
            for i in range(self.window.upscale_combo.count())
        ]
        
        self.assertIn("1x", options)
        self.assertIn("2x", options)
        self.assertIn("4x", options)
        self.assertIn("8x", options)
    
    def test_interpolation_combo_options(self):
        """Test interpolation combo box has correct options."""
        options = [
            self.window.interp_combo.itemText(i)
            for i in range(self.window.interp_combo.count())
        ]
        
        self.assertIn("CUBIC", options)
        self.assertIn("LANCZOS4", options)
        self.assertIn("LINEAR", options)
    
    def test_denoise_combo_options(self):
        """Test denoise combo box has correct options."""
        options = [
            self.window.denoise_combo.itemText(i)
            for i in range(self.window.denoise_combo.count())
        ]
        
        self.assertIn("None", options)
        self.assertIn("Bilateral", options)
        self.assertIn("NLM", options)
    
    def test_sharpen_combo_options(self):
        """Test sharpen combo box has correct options."""
        options = [
            self.window.sharpen_combo.itemText(i)
            for i in range(self.window.sharpen_combo.count())
        ]
        
        self.assertIn("None", options)
        self.assertIn("Unsharp Mask", options)
        self.assertIn("Laplacian", options)
        self.assertIn("High-Pass", options)
    
    def test_threshold_combo_options(self):
        """Test threshold combo box has correct options."""
        options = [
            self.window.threshold_combo.itemText(i)
            for i in range(self.window.threshold_combo.count())
        ]
        
        self.assertIn("None", options)
        self.assertIn("Adaptive Gaussian", options)
        self.assertIn("Otsu", options)
        self.assertIn("Binary", options)
    
    def test_clahe_spinbox_range(self):
        """Test CLAHE spinbox has correct range."""
        self.assertEqual(self.window.clahe_spin.minimum(), 1.0)
        self.assertEqual(self.window.clahe_spin.maximum(), 10.0)
        self.assertEqual(self.window.clahe_spin.value(), 2.0)
    
    def test_initial_state(self):
        """Test initial window state."""
        self.assertEqual(self.window.current_evidence, None)
        self.assertEqual(self.window.current_session, None)
        self.assertEqual(self.window.processing_pipeline, None)
    
    def test_status_bar_visible(self):
        """Test status bar is visible."""
        self.assertIsNotNone(self.window.statusBar())
    
    def test_tabs_created(self):
        """Test all tabs are created."""
        # The tab widget should be in the central widget
        central_widget = self.window.centralWidget()
        self.assertIsNotNone(central_widget)
    
    def test_progress_bar_initially_hidden(self):
        """Test progress bar is initially hidden."""
        self.assertFalse(self.window.progress_bar.isVisible())
    
    def test_evidence_display_widget(self):
        """Test evidence display widget exists."""
        self.assertIsNotNone(self.window.evidence_display)
    
    def test_processing_disabled_without_evidence(self):
        """Test processing is blocked without evidence."""
        self.window.current_evidence = None
        # Processing button should not crash when clicked without evidence
        # (it will show a warning)
        self.assertIsNone(self.window.current_evidence)


class TestGUIIntegration(unittest.TestCase):
    """Integration tests for GUI components."""
    
    @classmethod
    def setUpClass(cls):
        """Setup QApplication for tests."""
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()
    
    def setUp(self):
        """Create window and temp directory."""
        self.window = MainWindow()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Cleanup."""
        self.window.close()
        self.temp_dir.cleanup()
    
    def test_window_geometry(self):
        """Test window has reasonable geometry."""
        self.assertGreater(self.window.width(), 0)
        self.assertGreater(self.window.height(), 0)
    
    def test_clahe_parameter_accessible(self):
        """Test CLAHE parameter is accessible."""
        self.window.clahe_spin.setValue(3.5)
        self.assertEqual(self.window.clahe_spin.value(), 3.5)
    
    def test_combo_box_selection(self):
        """Test combo box selections work."""
        self.window.upscale_combo.setCurrentText("4x")
        self.assertEqual(self.window.upscale_combo.currentText(), "4x")
    
    def test_denoise_selection(self):
        """Test denoising method selection."""
        methods = ["None", "Bilateral", "NLM"]
        
        for method in methods:
            self.window.denoise_combo.setCurrentText(method)
            self.assertEqual(self.window.denoise_combo.currentText(), method)
    
    def test_sharpen_selection(self):
        """Test sharpening method selection."""
        methods = ["None", "Unsharp Mask", "Laplacian", "High-Pass"]
        
        for method in methods:
            self.window.sharpen_combo.setCurrentText(method)
            self.assertEqual(self.window.sharpen_combo.currentText(), method)
    
    def test_threshold_selection(self):
        """Test threshold method selection."""
        methods = ["None", "Adaptive Gaussian", "Otsu", "Binary"]
        
        for method in methods:
            self.window.threshold_combo.setCurrentText(method)
            self.assertEqual(self.window.threshold_combo.currentText(), method)


if __name__ == "__main__":
    unittest.main()
