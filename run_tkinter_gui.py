"""
Simple desktop GUI using Tkinter for forensic plate deblurring.
No networking required, just pure desktop application.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
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


class ForensicPlateGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 Forensic Plate Enhancer - Advanced Deblurring")
        self.root.geometry("1200x800")
        
        self.original_image = None
        self.current_image = None
        self.history = []
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the UI."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = ttk.LabelFrame(main_frame, text="🎛️  Controls", width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)
        
        # Upload button
        ttk.Button(left_panel, text="📥 Load License Plate Image", 
                   command=self.load_image).pack(fill=tk.X, padx=5, pady=5)
        
        # Method selection
        ttk.Label(left_panel, text="Select Deblurring Method:").pack(anchor=tk.W, padx=5, pady=(10, 5))
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
        
        # Apply button
        ttk.Button(left_panel, text="⚡ Apply Deblurring", 
                   command=self.apply_deblur).pack(fill=tk.X, padx=5, pady=10)
        
        # Enhancement options
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left_panel, text="Additional Enhancements:").pack(anchor=tk.W, padx=5)
        
        ttk.Button(left_panel, text="🎨 CLAHE Contrast", 
                   command=self.apply_clahe).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="🔤 Adaptive Threshold", 
                   command=self.apply_threshold).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="📈 Upscale 2x", 
                   command=self.upscale_2x).pack(fill=tk.X, padx=5, pady=2)
        
        # Control buttons
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(left_panel, text="↶ Undo", 
                   command=self.undo).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="⟲ Reset", 
                   command=self.reset).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(left_panel, text="💾 Save Result", 
                   command=self.save_image).pack(fill=tk.X, padx=5, pady=2)
        
        # Metrics
        ttk.Separator(left_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(left_panel, text="📊 Image Metrics:").pack(anchor=tk.W, padx=5)
        self.metrics_text = tk.Text(left_panel, height=8, width=35, font=("Courier", 9))
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel - Image display
        right_panel = ttk.LabelFrame(main_frame, text="🖼️  Image Display")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Image label
        self.image_label = ttk.Label(right_panel, text="No image loaded")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
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
            
            self.update_status(f"Loaded: {os.path.basename(file_path)}")
            self.display_image()
            self.update_metrics()
            
            # Get recommendation
            method, scores = recommend_deblur_method(self.original_image)
            self.method_var.set(method)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def display_image(self):
        """Display current image."""
        if self.current_image is None:
            return
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        # Resize for display
        h, w = rgb.shape[:2]
        if w > 600:
            scale = 600 / w
            rgb = cv2.resize(rgb, (600, int(h * scale)))
        
        # Convert to PhotoImage
        image_pil = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image_pil)
        
        # Update label
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo
    
    def update_metrics(self):
        """Update metrics display."""
        if self.current_image is None:
            self.metrics_text.delete(1.0, tk.END)
            return
        
        metrics = calculate_plate_visibility_score(self.current_image)
        
        text = f"""Visibility: {metrics['visibility_score']:.1f}/100
Sharpness: {metrics['sharpness']:.1f}
Contrast: {metrics['contrast']:.1f}
Edge Density: {metrics['edge_density']:.1f}
Texture Def: {metrics['texture_definition']:.0f}
Readable: {'Yes ✓' if metrics['is_readable'] else 'No ✗'}

Operations: {len(self.history)}
"""
        
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(1.0, text)
    
    def apply_deblur(self):
        """Apply deblurring."""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        method = self.method_var.get()
        if not method:
            messagebox.showwarning("Warning", "Please select a deblurring method")
            return
        
        self.update_status(f"Processing: {method}...")
        
        def process():
            try:
                image = self.current_image.copy()
                result = None
                
                if method == "blind_deconv":
                    result = apply_blind_deconvolution(image, iterations=20)
                elif method == "rl_motion":
                    result = apply_lucy_richardson_advanced(image, iterations=15, psf_type='motion')
                elif method == "rl_gaussian":
                    result = apply_lucy_richardson_advanced(image, iterations=15, psf_type='gaussian')
                elif method == "total_variation":
                    result = apply_total_variation_deblur(image, strength=0.1, iterations=100)
                elif method == "freq_domain":
                    result = apply_frequency_domain_deblur(image, blur_type='motion', strength=1.0)
                elif method == "super_res_2x":
                    result = apply_super_resolution_upscale(image, scale_factor=2, iterations=10)
                elif method == "super_res_4x":
                    result = apply_super_resolution_upscale(image, scale_factor=4, iterations=10)
                elif method == "multi_scale":
                    result = apply_multi_scale_deblur(image, scales=[0.5, 1.0])
                
                if result is not None:
                    self.history.append(self.current_image.copy())
                    self.current_image = result
                    
                    self.display_image()
                    self.update_metrics()
                    self.update_status(f"✓ Applied: {method}")
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
    print("FORENSIC PLATE ENHANCER - DESKTOP GUI")
    print("="*70)
    print("\n🖥️  Launching Tkinter GUI...\n")
    
    root = tk.Tk()
    app = ForensicPlateGUI(root)
    
    print("✓ GUI loaded successfully")
    print("✓ 9 deblurring methods available")
    print("✓ Real-time metrics display")
    print("✓ Full processing history with undo")
    print("\n⚠️  EXPERIMENTAL - Manual verification required\n")
    
    root.mainloop()
