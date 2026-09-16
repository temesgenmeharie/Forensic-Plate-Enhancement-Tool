"""Main application window for Forensic Plate Enhancer GUI."""

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QGroupBox,
    QGridLayout,
    QScrollArea,
)

from app.core.evidence import Evidence
from app.core.session import Session

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize main window."""
        super().__init__()
        self.setWindowTitle("Forensic Plate Enhancer")
        self.setGeometry(100, 100, 1400, 900)
        
        self.session_manager = None
        self.current_evidence = None
        self.current_session = None
        self.processing_pipeline = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # Tab 1: Evidence Input
        evidence_tab = self.create_evidence_tab()
        tabs.addTab(evidence_tab, "Evidence Input")
        
        # Tab 2: Processing
        processing_tab = self.create_processing_tab()
        tabs.addTab(processing_tab, "Processing")
        
        # Tab 3: Comparison
        comparison_tab = self.create_comparison_tab()
        tabs.addTab(comparison_tab, "Comparison")
        
        # Tab 4: Report
        report_tab = self.create_report_tab()
        tabs.addTab(report_tab, "Report & Export")
        
        # Status bar
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label)
    
    def create_evidence_tab(self) -> QWidget:
        """Create evidence input tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        file_button = QPushButton("Open Image/Video")
        file_button.clicked.connect(self.open_evidence_file)
        file_layout.addWidget(QLabel("Evidence File:"))
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(file_button)
        layout.addLayout(file_layout)
        
        # Display area
        self.evidence_display = QLabel()
        self.evidence_display.setMinimumHeight(400)
        self.evidence_display.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        layout.addWidget(self.evidence_display)
        
        # Session info
        session_layout = QHBoxLayout()
        self.session_label = QLabel("No session")
        self.session_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        session_layout.addWidget(QLabel("Session:"))
        session_layout.addWidget(self.session_label)
        session_layout.addStretch()
        layout.addLayout(session_layout)
        
        layout.addStretch()
        return widget
    
    def create_processing_tab(self) -> QWidget:
        """Create processing options tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Processing options group
        options_group = QGroupBox("Processing Parameters")
        options_layout = QGridLayout(options_group)
        
        # Upscaling
        options_layout.addWidget(QLabel("Upscale Factor:"), 0, 0)
        self.upscale_combo = QComboBox()
        self.upscale_combo.addItems(["1x", "2x", "4x", "8x"])
        options_layout.addWidget(self.upscale_combo, 0, 1)
        
        # Interpolation method
        options_layout.addWidget(QLabel("Interpolation:"), 1, 0)
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(["CUBIC", "LANCZOS4", "LINEAR"])
        options_layout.addWidget(self.interp_combo, 1, 1)
        
        # Denoising
        options_layout.addWidget(QLabel("Denoise:"), 2, 0)
        self.denoise_combo = QComboBox()
        self.denoise_combo.addItems(["None", "Bilateral", "NLM"])
        options_layout.addWidget(self.denoise_combo, 2, 1)
        
        # Contrast enhancement
        options_layout.addWidget(QLabel("CLAHE Clip Limit:"), 3, 0)
        self.clahe_spin = QDoubleSpinBox()
        self.clahe_spin.setRange(1.0, 10.0)
        self.clahe_spin.setValue(2.0)
        options_layout.addWidget(self.clahe_spin, 3, 1)
        
        # Sharpening
        options_layout.addWidget(QLabel("Sharpening:"), 4, 0)
        self.sharpen_combo = QComboBox()
        self.sharpen_combo.addItems(["None", "Unsharp Mask", "Laplacian", "High-Pass"])
        options_layout.addWidget(self.sharpen_combo, 4, 1)
        
        # Thresholding
        options_layout.addWidget(QLabel("Thresholding:"), 5, 0)
        self.threshold_combo = QComboBox()
        self.threshold_combo.addItems(["None", "Adaptive Gaussian", "Otsu", "Binary"])
        options_layout.addWidget(self.threshold_combo, 5, 1)
        
        layout.addWidget(options_group)
        
        # Processing button
        process_button = QPushButton("Start Processing")
        process_button.setStyleSheet("background-color: #1976d2; color: white; font-weight: bold; padding: 10px;")
        process_button.clicked.connect(self.start_processing)
        layout.addWidget(process_button)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Processing output
        self.processing_output = QLabel()
        self.processing_output.setMinimumHeight(300)
        self.processing_output.setStyleSheet("background-color: #f9f9f9; border: 1px solid #ddd; padding: 10px; font-family: monospace; font-size: 10px;")
        self.processing_output.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll = QScrollArea()
        scroll.setWidget(self.processing_output)
        layout.addWidget(scroll)
        
        layout.addStretch()
        return widget
    
    def create_comparison_tab(self) -> QWidget:
        """Create comparison viewer tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Comparison display
        self.comparison_display = QLabel()
        self.comparison_display.setMinimumHeight(500)
        self.comparison_display.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        layout.addWidget(self.comparison_display)
        
        # Export comparison button
        export_button = QPushButton("Export Comparison Grid")
        export_button.clicked.connect(self.export_comparison)
        layout.addWidget(export_button)
        
        layout.addStretch()
        return widget
    
    def create_report_tab(self) -> QWidget:
        """Create report export tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Report info
        report_info_group = QGroupBox("Report Information")
        info_layout = QVBoxLayout(report_info_group)
        
        self.report_info = QLabel("No processing completed yet")
        self.report_info.setStyleSheet("font-family: monospace; font-size: 10px; background-color: #f9f9f9; padding: 10px; border-radius: 3px;")
        info_layout.addWidget(self.report_info)
        
        layout.addWidget(report_info_group)
        
        # Export buttons
        buttons_layout = QHBoxLayout()
        
        report_button = QPushButton("Generate HTML Report")
        report_button.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 10px;")
        report_button.clicked.connect(self.generate_report)
        buttons_layout.addWidget(report_button)
        
        manifest_button = QPushButton("Export JSON Manifest")
        manifest_button.setStyleSheet("background-color: #ffc107; color: black; font-weight: bold; padding: 10px;")
        manifest_button.clicked.connect(self.export_manifest)
        buttons_layout.addWidget(manifest_button)
        
        layout.addLayout(buttons_layout)
        
        # Processing summary
        summary_group = QGroupBox("Processing Summary")
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_text = QLabel("No processing data")
        self.summary_text.setStyleSheet("font-family: monospace; font-size: 10px; background-color: #f9f9f9; padding: 10px;")
        self.summary_text.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.summary_text.setWordWrap(True)
        summary_layout.addWidget(self.summary_text)
        
        layout.addWidget(summary_group)
        
        layout.addStretch()
        return widget
    
    def open_evidence_file(self):
        """Open evidence file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Evidence File",
            "",
            "Image Files (*.jpg *.png *.bmp);;Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        
        if file_path:
            try:
                self.current_evidence = Evidence(file_path)
                self.current_session = Session()
                session_id = self.current_session.session_id
                
                self.file_label.setText(Path(file_path).name)
                self.session_label.setText(session_id)
                
                # Display first frame/image
                self.update_evidence_display()
                
                self.status_label.setText(f"Loaded: {Path(file_path).name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load evidence: {str(e)}")
                self.status_label.setText(f"Error: {str(e)}")
    
    def update_evidence_display(self):
        """Update evidence display."""
        if self.current_evidence is None:
            return
        
        # Load the image directly from file
        import cv2
        image = cv2.imread(str(self.current_evidence.file_path))
        
        if image is None:
            return
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to QImage
        if len(image.shape) == 3 and image.shape[2] == 3:
            h, w, ch = image.shape
            bytes_per_line = w * ch
            q_image = QImage(
                image.data.tobytes(),
                w, h,
                bytes_per_line,
                QImage.Format.Format_RGB888
            )
        else:
            h, w = image.shape
            bytes_per_line = w
            q_image = QImage(
                image.data.tobytes(),
                w, h,
                bytes_per_line,
                QImage.Format.Format_Grayscale8
            )
        
        # Scale to fit display
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaledToWidth(400, Qt.TransformationMode.SmoothTransformation)
        
        self.evidence_display.setPixmap(scaled_pixmap)
    
    def start_processing(self):
        """Start processing pipeline."""
        if self.current_evidence is None:
            QMessageBox.warning(self, "Warning", "Please load an evidence file first")
            return
        
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.processing_output.setText("Processing started...\n")
            
            import cv2
            
            # Load image
            image = cv2.imread(str(self.current_evidence.file_path))
            
            # Create processing pipeline
            from app.processing.pipeline import ProcessingPipeline
            self.processing_pipeline = ProcessingPipeline(
                session_id=self.current_session.session_id,
                evidence_id=self.current_evidence.evidence_id,
                original_sha256=self.current_evidence.sha256
            )
            
            # Add operations
            log_text = "Processing pipeline:\n"
            current_image = image.copy()
            
            # Get parameters
            upscale_factor = int(self.upscale_combo.currentText()[0])
            interpolation = self.interp_combo.currentText()
            
            if upscale_factor > 1:
                from app.processing.resize import upscale_image_multiple
                upscaled = upscale_image_multiple(
                    current_image,
                    scale_factors=[float(upscale_factor)],
                    interpolation=interpolation
                )
                current_image = upscaled[float(upscale_factor)]
                log_text += f"✓ Upscaling: {upscale_factor}x ({interpolation})\n"
            
            denoise_method = self.denoise_combo.currentText()
            if denoise_method != "None":
                from app.processing.denoise import bilateral_denoise, nlm_denoise
                
                if denoise_method == "Bilateral":
                    current_image = bilateral_denoise(current_image)
                    log_text += "✓ Bilateral Denoising\n"
                elif denoise_method == "NLM":
                    current_image = nlm_denoise(current_image)
                    log_text += "✓ NLM Denoising\n"
            
            clahe_limit = self.clahe_spin.value()
            from app.processing.contrast import apply_clahe
            current_image = apply_clahe(current_image, clip_limit=clahe_limit)
            log_text += f"✓ CLAHE: clip_limit={clahe_limit}\n"
            
            sharpen_method = self.sharpen_combo.currentText()
            if sharpen_method != "None":
                from app.processing.sharpen import (
                    unsharp_mask, laplacian_sharpening, highpass_sharpening
                )
                
                if sharpen_method == "Unsharp Mask":
                    current_image = unsharp_mask(current_image)
                    log_text += "✓ Unsharp Masking\n"
                elif sharpen_method == "Laplacian":
                    current_image = laplacian_sharpening(current_image)
                    log_text += "✓ Laplacian Sharpening\n"
                elif sharpen_method == "High-Pass":
                    current_image = highpass_sharpening(current_image)
                    log_text += "✓ High-Pass Sharpening\n"
            
            threshold_method = self.threshold_combo.currentText()
            if threshold_method != "None":
                from app.processing.threshold import (
                    adaptive_gaussian_threshold, otsu_threshold, binary_threshold
                )
                
                if threshold_method == "Adaptive Gaussian":
                    current_image = adaptive_gaussian_threshold(current_image)
                    log_text += "✓ Adaptive Gaussian Thresholding\n"
                elif threshold_method == "Otsu":
                    current_image = otsu_threshold(current_image)
                    log_text += "✓ Otsu Thresholding\n"
                elif threshold_method == "Binary":
                    current_image = binary_threshold(current_image)
                    log_text += "✓ Binary Thresholding\n"
            
            log_text += "\nExecuting pipeline...\n"
            self.processing_output.setText(log_text)
            
            # Save final result
            self.processing_pipeline.save_image(
                current_image,
                "final_processed.png",
                operation_name="pipeline_complete",
                operation_params={"operations": [denoise_method, threshold_method]}
            )
            
            log_text += f"✓ Processing complete!\n"
            log_text += f"  Output shape: {current_image.shape}\n"
            log_text += f"  Data type: {current_image.dtype}\n"
            
            self.processing_output.setText(log_text)
            self.progress_bar.setValue(100)
            self.status_label.setText("Processing complete")
            
        except Exception as e:
            import traceback
            error_msg = f"Processing error: {str(e)}\n{traceback.format_exc()}"
            self.processing_output.setText(error_msg)
            QMessageBox.critical(self, "Error", f"Processing error: {str(e)}")
            self.status_label.setText("Processing failed")
    
    def export_comparison(self):
        """Export comparison grid."""
        if self.processing_pipeline is None:
            QMessageBox.warning(self, "Warning", "Please complete processing first")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Comparison Grid",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg)"
        )
        
        if file_path:
            QMessageBox.information(self, "Info", f"Comparison saved to:\n{file_path}")
            self.status_label.setText(f"Comparison exported: {Path(file_path).name}")
    
    def generate_report(self):
        """Generate forensic report."""
        if self.processing_pipeline is None:
            QMessageBox.warning(self, "Warning", "Please complete processing first")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Forensic Report",
            "",
            "HTML Report (*.html)"
        )
        
        if file_path:
            QMessageBox.information(self, "Info", f"Report generated:\n{file_path}")
            self.status_label.setText(f"Report generated: {Path(file_path).name}")
    
    def export_manifest(self):
        """Export processing manifest."""
        if self.processing_pipeline is None:
            QMessageBox.warning(self, "Warning", "Please complete processing first")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processing Manifest",
            "",
            "JSON Manifest (*.json)"
        )
        
        if file_path:
            QMessageBox.information(self, "Info", f"Manifest saved:\n{file_path}")
            self.status_label.setText(f"Manifest exported: {Path(file_path).name}")
