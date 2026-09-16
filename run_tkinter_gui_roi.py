"""
Enhanced Tkinter GUI with ROI (Region of Interest) selection.
Allows users to select specific rectangular areas for targeted deblurring.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
from threading import Thread
import os

from app.processing.advanced_deblur import (
    apply_blind_deconvolution,
    apply_lucy_richardson_advanced,
    apply_total_variation_deblur,
    apply_frequency_domain_deblur,
    apply_super_resolution_upscale,
    apply_multi_scale_deblur,
    calculate_plate_visibility_score,
    recommend_deblur_method
)
from app.processing.resize import upscale_image_multiple
from app.processing.contrast import apply_clahe
from app.processing.threshold import apply_adaptive_gaussian_threshold


class ForensicPlateGUIWithROI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 Forensic Plate Enhancer - ROI Selection & Advanced Deblurring")
        self.root.geometry("1400x900")
        
        self.original_image = None
        self.current_image = None
        self.display_image_pil = None
        self.history = []
        
        # ROI selection variables
        self.roi_mode = False
        self.roi_start = None
        self.roi_end = None
        self.selected_roi = None
        self.roi_canvas_image = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI with ROI selection."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = ttk.LabelFrame(main_frame, text="🎛️  Controls & ROI", width=320)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)
        
        # Upload button
        ttk.Button(left_panel, text="📥 Load License Plate Image", 
                   command=self.load_image).pack(fill=tk.X, padx=5, pady=5)
        
        # ROI Selection Section
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left_panel, text="🎯 ROI Selection:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=5)
        
        roi_info = """1. Click "Select ROI Area"
2. Drag rectangle on image
3. Choose to process ROI or
   Full image"""
        ttk.Label(left_panel, text=roi_info, font=("Courier", 8), 
                 justify=tk.LEFT).pack(anchor=tk.W, padx=10, pady=5)
        
        ttk.Button(left_panel, text="🎯 Select ROI Area", 
                   command=self.start_roi_selection).pack(fill=tk.X, padx=5, pady=5)
        
        self.roi_status = tk.StringVar(value="No ROI selected")
        ttk.Label(left_panel, textvariable=self.roi_status, 
                 font=("Courier", 8)).pack(anchor=tk.W, padx=5, pady=2)
        
        # Method selection
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left_panel, text="⚡ Deblurring Method:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=5)
        
        self.method_var = tk.StringVar()
        methods = [
            ("Blind Deconvolution", "blind_deconv"),
            ("Richardson-Lucy (Motion)", "rl_motion"),
            ("Richardson-Lucy (Gaussian)", "rl_gaussian"),
            ("Total Variation", "total_variation"),
            ("Frequency Domain", "freq_domain"),
            ("Super-Resolution 2x", "super_res_2x"),
            ("Super-Resolution 4x", "super_res_4x"),
            ("Multi-Scale", "multi_scale"),
        ]
        
        for text, value in methods:
            ttk.Radiobutton(left_panel, text=text, variable=self.method_var, 
                           value=value).pack(anchor=tk.W, padx=15, pady=2)
        
        # Processing target
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left_panel, text="🎯 Process Target:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=5)
        
        self.target_var = tk.StringVar(value="full")
        ttk.Radiobutton(left_panel, text="Full Image", variable=self.target_var, 
                       value="full").pack(anchor=tk.W, padx=15, pady=2)
        ttk.Radiobutton(left_panel, text="Selected ROI Only", variable=self.target_var, 
                       value="roi").pack(anchor=tk.W, padx=15, pady=2)
        
        # Apply button
        ttk.Button(left_panel, text="⚡ Apply Deblurring", 
                   command=self.apply_deblur).pack(fill=tk.X, padx=5, pady=10)
        
        # Enhancement options
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left_panel, text="🎨 Enhancements:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=5)
        
        ttk.Button(left_panel, text="CLAHE Contrast", 
                   command=self.apply_clahe).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="Adaptive Threshold", 
                   command=self.apply_threshold).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="Upscale 2x", 
                   command=self.upscale_2x).pack(fill=tk.X, padx=5, pady=2)
        
        # Control buttons
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        button_frame = ttk.Frame(left_panel)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="↶ Undo", 
                   command=self.undo).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(button_frame, text="⟲ Reset", 
                   command=self.reset).pack(side=tk.LEFT, expand=True, padx=2)
        ttk.Button(button_frame, text="💾 Save", 
                   command=self.save_image).pack(side=tk.LEFT, expand=True, padx=2)
        
        # Metrics
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left_panel, text="📊 Metrics:", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=5)
        self.metrics_text = tk.Text(left_panel, height=6, width=38, font=("Courier", 8))
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel - Image display with canvas for ROI
        right_panel = ttk.LabelFrame(main_frame, text="🖼️  Image Display & ROI Selection")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Canvas for image and ROI drawing
        self.canvas = tk.Canvas(right_panel, bg="gray", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=5, pady=5)
    
    def update_status(self, message):
        """Update status bar."""
        self.status_var.set(message)
        self.root.update()
    
    def load_image(self):
        """Load image file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.original_image = cv2.imread(file_path)
            self.current_image = self.original_image.copy()
            self.history = []
            self.selected_roi = None
            self.roi_mode = False
            
            self.update_status(f"Loaded: {os.path.basename(file_path)}")
            self.display_image()
            self.update_metrics()
            
            # Get recommendation
            method, scores = recommend_deblur_method(self.original_image)
            self.method_var.set(method)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def display_image(self):
        """Display current image on canvas."""
        if self.current_image is None:
            return
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # Create PIL image
        self.display_image_pil = Image.fromarray(rgb)
        
        # Draw ROI if selected
        if self.selected_roi:
            draw = ImageDraw.Draw(self.display_image_pil)
            x1, y1, x2, y2 = self.selected_roi
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Resize for canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width < 2 or canvas_height < 2:
            canvas_width = 600
            canvas_height = 400
        
        self.display_image_pil_resized = self.display_image_pil.copy()
        self.display_image_pil_resized.thumbnail((canvas_width - 10, canvas_height - 10), Image.Resampling.LANCZOS)
        
        # Store scale factors for ROI coordinates
        self.scale_x = self.display_image_pil.width / self.display_image_pil_resized.width
        self.scale_y = self.display_image_pil.height / self.display_image_pil_resized.height
        
        photo = ImageTk.PhotoImage(self.display_image_pil_resized)
        
        self.canvas.delete("all")
        self.canvas_image = self.canvas.create_image(
            canvas_width // 2, canvas_height // 2, image=photo
        )
        self.canvas.image = photo
    
    def start_roi_selection(self):
        """Start ROI selection mode."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        self.roi_mode = True
        self.roi_start = None
        self.roi_end = None
        self.update_status("🎯 Draw rectangle on image to select ROI")
    
    def on_canvas_click(self, event):
        """Handle canvas click."""
        if not self.roi_mode or self.current_image is None:
            return
        
        self.roi_start = (event.x, event.y)
    
    def on_canvas_drag(self, event):
        """Handle canvas drag."""
        if not self.roi_mode or self.roi_start is None:
            return
        
        self.roi_end = (event.x, event.y)
        self.redraw_canvas_with_preview()
    
    def on_canvas_release(self, event):
        """Handle canvas release."""
        if not self.roi_mode or self.roi_start is None:
            return
        
        self.roi_end = (event.x, event.y)
        
        # Convert canvas coordinates to image coordinates
        x1_canvas, y1_canvas = self.roi_start
        x2_canvas, y2_canvas = self.roi_end
        
        x1 = int(min(x1_canvas, x2_canvas) * self.scale_x)
        y1 = int(min(y1_canvas, y2_canvas) * self.scale_y)
        x2 = int(max(x1_canvas, x2_canvas) * self.scale_x)
        y2 = int(max(y1_canvas, y2_canvas) * self.scale_y)
        
        # Clamp to image bounds
        h, w = self.current_image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x2 > x1 and y2 > y1:
            self.selected_roi = (x1, y1, x2, y2)
            self.roi_status.set(f"ROI: {x1},{y1} -> {x2},{y2} ({x2-x1}x{y2-y1})")
            self.update_status(f"✓ ROI selected: {x2-x1}x{y2-y1} pixels")
        
        self.roi_mode = False
        self.display_image()
    
    def redraw_canvas_with_preview(self):
        """Redraw canvas with ROI preview."""
        if self.display_image_pil is None:
            return
        
        display_copy = self.display_image_pil.copy()
        draw = ImageDraw.Draw(display_copy)
        
        if self.roi_start and self.roi_end:
            x1 = min(self.roi_start[0], self.roi_end[0])
            y1 = min(self.roi_start[1], self.roi_end[1])
            x2 = max(self.roi_start[0], self.roi_end[0])
            y2 = max(self.roi_start[1], self.roi_end[1])
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=2)
        
        display_resized = display_copy.copy()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            display_resized.thumbnail((canvas_width - 10, canvas_height - 10), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(display_resized)
        
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo)
        self.canvas.image = photo
    
    def update_metrics(self):
        """Update metrics display."""
        if self.current_image is None:
            self.metrics_text.delete(1.0, tk.END)
            return
        
        metrics = calculate_plate_visibility_score(self.current_image)
        
        # Calculate ROI metrics if selected
        roi_text = ""
        if self.selected_roi:
            x1, y1, x2, y2 = self.selected_roi
            roi_image = self.current_image[y1:y2, x1:x2]
            roi_metrics = calculate_plate_visibility_score(roi_image)
            roi_text = f"\nROI Visibility: {roi_metrics['visibility_score']:.1f}/100"
        
        text = f"""Full Image:
Visibility: {metrics['visibility_score']:.1f}/100
Sharpness: {metrics['sharpness']:.1f}
Contrast: {metrics['contrast']:.1f}
Edge Density: {metrics['edge_density']:.1f}
Readable: {'✓' if metrics['is_readable'] else '✗'}{roi_text}

Operations: {len(self.history)}
"""
        
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, text)
    
    def apply_deblur(self):
        """Apply deblurring to full image or ROI."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        method = self.method_var.get()
        if not method:
            messagebox.showwarning("Warning", "Please select a deblurring method")
            return
        
        target = self.target_var.get()
        
        if target == "roi" and self.selected_roi is None:
            messagebox.showwarning("Warning", "Please select a ROI area first")
            return
        
        self.update_status(f"Processing: {method}...")
        
        def process():
            try:
                image = self.current_image.copy()
                result = None
                
                # Extract ROI if needed
                if target == "roi":
                    x1, y1, x2, y2 = self.selected_roi
                    roi_image = image[y1:y2, x1:x2].copy()
                else:
                    roi_image = image
                
                # Apply deblurring
                if method == "blind_deconv":
                    result = apply_blind_deconvolution(roi_image, iterations=20)
                elif method == "rl_motion":
                    result = apply_lucy_richardson_advanced(roi_image, iterations=15, psf_type='motion')
                elif method == "rl_gaussian":
                    result = apply_lucy_richardson_advanced(roi_image, iterations=15, psf_type='gaussian')
                elif method == "total_variation":
                    result = apply_total_variation_deblur(roi_image, strength=0.1, iterations=100)
                elif method == "freq_domain":
                    result = apply_frequency_domain_deblur(roi_image, blur_type='motion', strength=1.0)
                elif method == "super_res_2x":
                    result = apply_super_resolution_upscale(roi_image, scale_factor=2, iterations=10)
                elif method == "super_res_4x":
                    result = apply_super_resolution_upscale(roi_image, scale_factor=4, iterations=10)
                elif method == "multi_scale":
                    result = apply_multi_scale_deblur(roi_image, scales=[0.5, 1.0])
                
                if result is not None:
                    # Replace ROI in image if processing ROI
                    if target == "roi":
                        # Handle size mismatch for super-resolution
                        if result.shape[:2] != (y2-y1, x2-x1):
                            result = cv2.resize(result, (x2-x1, y2-y1))
                        image[y1:y2, x1:x2] = result
                    else:
                        image = result
                    
                    self.history.append(self.current_image.copy())
                    self.current_image = image
                    
                    self.display_image()
                    self.update_metrics()
                    self.update_status(f"✓ Applied: {method} on {target}")
                else:
                    self.update_status("Error: Processing failed")
            
            except Exception as e:
                messagebox.showerror("Error", f"Processing failed: {e}")
                self.update_status("Error")
        
        Thread(target=process, daemon=True).start()
    
    def apply_clahe(self):
        """Apply CLAHE."""
        if self.current_image is None:
            return
        
        self.history.append(self.current_image.copy())
        self.current_image = apply_clahe(self.current_image, clip_limit=2.0)
        
        self.display_image()
        self.update_metrics()
        self.update_status("✓ CLAHE applied")
    
    def apply_threshold(self):
        """Apply threshold."""
        if self.current_image is None:
            return
        
        self.history.append(self.current_image.copy())
        self.current_image = apply_adaptive_gaussian_threshold(self.current_image)
        
        self.display_image()
        self.update_metrics()
        self.update_status("✓ Threshold applied")
    
    def upscale_2x(self):
        """Upscale 2x."""
        if self.current_image is None:
            return
        
        self.history.append(self.current_image.copy())
        upscaled = upscale_image_multiple(self.current_image, scale_factors=[2.0])
        self.current_image = upscaled[2.0]
        
        self.display_image()
        self.update_metrics()
        self.update_status("✓ Upscaled 2x")
    
    def undo(self):
        """Undo last operation."""
        if self.history:
            self.current_image = self.history.pop()
            self.display_image()
            self.update_metrics()
            self.update_status("↶ Undo")
        else:
            messagebox.showinfo("Info", "Nothing to undo")
    
    def reset(self):
        """Reset to original."""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
            self.history = []
            self.display_image()
            self.update_metrics()
            self.update_status("⟲ Reset to original")
    
    def save_image(self):
        """Save processed image."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("All", "*.*")]
        )
        
        if file_path:
            cv2.imwrite(file_path, self.current_image)
            messagebox.showinfo("Success", f"Image saved:\n{file_path}")
            self.update_status(f"✓ Saved: {os.path.basename(file_path)}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FORENSIC PLATE ENHANCER - ROI SELECTION & ADVANCED DEBLURRING")
    print("="*70)
    print("\n🖥️  Launching Enhanced Tkinter GUI...\n")
    
    root = tk.Tk()
    app = ForensicPlateGUIWithROI(root)
    
    print("✓ GUI loaded with ROI selection")
    print("✓ 9 deblurring methods available")
    print("✓ Targeted ROI processing")
    print("✓ Real-time metrics & undo history")
    print("\n⚠️  EXPERIMENTAL - Manual verification required\n")
    
    root.mainloop()
