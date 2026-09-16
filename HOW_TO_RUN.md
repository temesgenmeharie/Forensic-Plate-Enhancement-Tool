# How to Run - Forensic Plate Enhancer

## Quick Start

### Option 1: Desktop GUI with ROI Selection (Recommended) ⭐

```bash
cd d:\JSK\FIDT
python run_tkinter_gui_roi.py
```

**Features:**
- 🎯 Rectangular ROI selection for targeted deblurring
- 9 advanced deblurring methods
- Real-time visibility metrics
- Full undo/reset history
- Save processed images

**Steps:**
1. Click "📥 Load License Plate Image"
2. Click "🎯 Select ROI Area" and drag rectangle on image
3. Choose deblurring method
4. Choose target: "Full Image" or "Selected ROI Only"
5. Click "⚡ Apply Deblurring"
6. Stack multiple operations (CLAHE, threshold, etc)
7. Click "💾 Save" to export

---

### Option 2: Basic Desktop GUI

```bash
cd d:\JSK\FIDT
python run_tkinter_gui.py
```

**Features:**
- Full image deblurring
- 9 methods available
- Real-time metrics
- Undo/reset

---

### Option 3: Interactive Demo (Command Line)

```bash
cd d:\JSK\FIDT
python demo_deblurring.py
```

Shows 4 comprehensive demonstrations:
- DEMO 1: Basic deblurring techniques
- DEMO 2: Super-resolution for small plates
- DEMO 3: Combined pipeline
- DEMO 4: Method comparison

---

## System Requirements

✅ Python 3.8+  
✅ OpenCV 4.8.1  
✅ Pillow (PIL)  
✅ NumPy, SciPy  
✅ Tkinter (included with Python)

## Installation

```bash
# Navigate to project
cd d:\JSK\FIDT

# Install dependencies (if needed)
pip install -r requirements.txt
```

## 9 Available Deblurring Methods

1. **Blind Deconvolution** - Best for unknown blur types
2. **Richardson-Lucy (Motion)** - Motion blur removal
3. **Richardson-Lucy (Gaussian)** - Out-of-focus blur
4. **Total Variation** - Edge-preserving deblurring
5. **Frequency Domain** - Fast, blur-type specific
6. **Super-Resolution 2x** - 2x upscaling with refinement
7. **Super-Resolution 4x** - 4x upscaling (extreme cases)
8. **Multi-Scale** - Multi-scale processing
9. **Morphological Enhancement** - Text clarity

## GUI Shortcuts & Tips

### Keyboard Controls
- **Undo** (↶): Revert last operation
- **Reset** (⟲): Back to original
- **Save** (💾): Export image

### ROI Tips
- Drag to create rectangle
- Release to confirm selection
- View coordinates in status bar
- ROI shown in red outline
- Switch between "Full Image" and "ROI Only" processing

### Processing Tips
- **For motion blur**: Use Richardson-Lucy (Motion)
- **For out-of-focus**: Use Richardson-Lucy (Gaussian)
- **For unknown blur**: Use Blind Deconvolution
- **For small plates**: Use Super-Resolution 4x
- **For text clarity**: Use Total Variation + Morphological

## Example Workflows

### Workflow 1: Motion-Blurred Plate
```
1. Load image
2. Select plate region (ROI)
3. Richardson-Lucy (Motion)
4. Super-Resolution 2x
5. CLAHE Contrast
6. Adaptive Threshold
7. Save
```

### Workflow 2: Out-of-Focus Plate
```
1. Load image
2. Richardson-Lucy (Gaussian)
3. Total Variation
4. CLAHE Contrast
5. Save
```

### Workflow 3: Very Small/Pixelated Plate
```
1. Load image
2. Super-Resolution 4x
3. Blind Deconvolution
4. CLAHE Contrast
5. Morphological Enhancement
6. Save
```

## Metrics Explained

- **Visibility Score** (0-100): Overall plate readability
- **Sharpness**: Edge definition quality
- **Contrast**: Difference between light/dark areas
- **Edge Density**: Amount of detected edges
- **Readable**: Binary yes/no indicator
- **Texture Definition**: Character clarity count

## Important Notes

⚠️ **EXPERIMENTAL**: All deblurring is marked EXPERIMENTAL  
⚠️ **Manual Verification**: Results require examiner review  
⚠️ **Evidence Preservation**: Original never modified  
⚠️ **Audit Trail**: All operations logged  

## Troubleshooting

### GUI doesn't open
- Ensure Python 3.8+ is installed
- Check all dependencies: `pip install -r requirements.txt`
- Run from correct directory: `cd d:\JSK\FIDT`

### Image won't load
- Verify file is valid JPG/PNG/BMP
- File should be 200+ pixels wide
- Try different image first

### Processing too slow
- Use "Frequency Domain" method (fast)
- Reduce image size first
- Use "Selected ROI Only" to process smaller area

### Results look worse
- Try different deblurring method
- Reduce iteration counts
- Check if image is already clear

## Example Test Images

Create test plate:
```python
import cv2
import numpy as np

# Yellow license plate
plate = np.ones((100, 250, 3), dtype=np.uint8) * 220
plate[:, :, :2] = 0

# Add text
cv2.putText(plate, "ABC-1234", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 
            1.8, (0, 0, 0), 3)

# Blur it
blurred = cv2.GaussianBlur(plate, (21, 21), 3.0)

# Save
cv2.imwrite("test_plate.jpg", blurred)
```

Then open `test_plate.jpg` in GUI!

## Support

For detailed information:
- See `ADVANCED_DEBLURRING_GUIDE.md` for technical details
- See `README.md` for full documentation
- See `PHASE_SUMMARY.md` for project overview

## Getting Started

**Fastest way to test:**

```bash
cd d:\JSK\FIDT
python run_tkinter_gui_roi.py
```

A window will open. Load an image and start deblurring! 🎯

---

**Version**: 1.0.0  
**Status**: Ready for use  
⚠️ For forensic investigation use only
