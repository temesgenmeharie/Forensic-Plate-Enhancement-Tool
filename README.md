# Forensic License Plate Image & Video Enhancement Tool

A professional Python application for forensic analysis of license plate images and videos. This tool is designed for legitimate digital-forensics investigations and emphasizes evidence preservation, integrity verification, and comprehensive audit trails.

## Project Overview

The application provides a structured workflow for:

- **Evidence Preservation**: Original files are never modified; all processing creates derivatives
- **Integrity Verification**: SHA-256 hashing of all evidence and processed files
- **License Plate Analysis**: Cropping, upscaling, enhancement, and comparison
- **Video Processing**: Frame extraction and multi-frame analysis
- **Forensic Reporting**: Comprehensive HTML reports with full audit trails
- **Metadata Management**: Complete processing history and parameters

## Technology Stack

- **Python**: 3.12+
- **Image Processing**: OpenCV 4.8+
- **Numerical Computing**: NumPy, SciPy
- **GUI**: PySide6 (Qt for Python)
- **Media Handling**: Pillow, FFmpeg
- **Testing**: unittest (standard library)

## Project Structure

```
forensic-plate-enhancer/
│
├── app/
│   ├── __init__.py
│   ├── logging_config.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── evidence.py          # Evidence intake & metadata
│   │   ├── hashing.py           # SHA-256 integrity verification
│   │   ├── session.py           # Session management
│   │   └── metadata.py          # Processing metadata & manifests
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── crop.py              # Plate region cropping
│   │   ├── resize.py            # Upscaling (Phase 2)
│   │   ├── grayscale.py         # Grayscale conversion (Phase 3)
│   │   ├── contrast.py          # CLAHE enhancement (Phase 3)
│   │   ├── denoise.py           # Denoising filters (Phase 3)
│   │   ├── sharpen.py           # Sharpening (Phase 3)
│   │   ├── threshold.py         # Adaptive thresholding (Phase 3)
│   │   ├── deblur.py            # Deblurring methods (Phase 4)
│   │   └── pipeline.py          # Processing pipeline
│   │
│   ├── video/
│   │   ├── __init__.py
│   │   ├── reader.py            # Video reading (Phase 5)
│   │   ├── frame_extractor.py   # Frame extraction (Phase 5)
│   │   ├── frame_quality.py     # Quality metrics (Phase 5)
│   │   └── frame_alignment.py   # Frame alignment (Phase 6)
│   │
│   ├── comparison/
│   │   ├── __init__.py
│   │   └── comparison.py        # Comparison interface (Phase 7)
│   │
│   └── reporting/
│       ├── __init__.py
│       ├── report.py            # HTML report generation (Phase 8)
│       └── manifest.py          # Processing manifest (Phase 8)
│
├── tests/
│   ├── __init__.py
│   ├── test_hashing.py
│   ├── test_session.py
│   ├── test_evidence.py
│   └── test_crop.py
│
├── input/                       # Place evidence files here
├── output/                      # Processed derivatives
├── logs/                        # Application logs
├── requirements.txt
├── README.md
└── main.py
```

## Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager
- FFmpeg (for video processing)

### Setup

1. Clone or extract the project:
```bash
cd forensic-plate-enhancer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create necessary directories:
```bash
mkdir -p input output logs
```

## Usage

### Phase 1: Evidence Intake (Current)

To demonstrate Phase 1 functionality:

1. Place evidence files in the `input/` directory:
   - Images: `.jpg`, `.jpeg`, `.png`, `.tiff`, `.bmp`
   - Videos: `.mp4`, `.avi`, `.mov`, `.mkv`

2. Run the main script:
```bash
python main.py
```

This will:
- Create a unique analysis session
- Load evidence files
- Calculate SHA-256 hashes
- Extract and display metadata
- Create processing metadata records

### Running Tests

Run all unit tests:
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Run specific test module:
```bash
python -m unittest tests.test_hashing -v
python -m unittest tests.test_evidence -v
python -m unittest tests.test_crop -v
```

## Phase 1: Implementation Details

### Evidence Management

The `Evidence` class handles:
- File validation (format, accessibility)
- SHA-256 integrity hashing
- Metadata extraction (dimensions, codec, duration, etc.)
- Session tracking

**Supported Formats:**
- Images: JPEG, PNG, TIFF, BMP
- Videos: MP4, AVI, MOV, MKV

### SHA-256 Hashing

All evidence and derivatives are hashed using SHA-256 for integrity verification:
- Original evidence hash recorded at intake
- All processed derivatives hashed
- Hash verification available

### Session Management

Each analysis creates a unique session with:
- Unique session ID (format: `SES-YYYY-XXXXXX`)
- Creation timestamp
- Evidence tracking
- Operation history

### Metadata Recording

Processing metadata includes:
- Original file information
- SHA-256 hash of original
- Processing start time
- All operations performed
- Parameters for each operation
- Output file hashes
- Software version

## Forensic Requirements

### Evidence Preservation

The application enforces strict evidence preservation:

```
Original Evidence
       │
       ├── SHA-256 hash (recorded)
       │
       └── Read-only source
               │
               ▼
       Processing Pipeline
               │
               ▼
       Processed Derivatives
       (tracked and hashed)
```

**Key Principles:**
- Original files are NEVER modified
- All processing creates new derivative files
- SHA-256 hash of original is recorded and can be verified
- Complete audit trail maintained
- Every derivative is traceable to original evidence

### Metadata Structure

Evidence metadata JSON example:
```json
{
  "evidence_id": "EV-20260916153022-1234",
  "original_filename": "cctv_vehicle_001.mp4",
  "file_extension": ".mp4",
  "file_size": 524288000,
  "sha256": "a1b2c3d4e5f6...",
  "type": "video",
  "resolution": "1920x1080",
  "intake_timestamp": "2026-09-16T15:30:22.123456",
  "width": 1920,
  "height": 1080,
  "fps": 29.97,
  "frame_count": 86400,
  "duration": 2884.57,
  "codec": "h264",
  "software_version": "1.0.0"
}
```

## Development Phases

The project is being implemented incrementally:

### ✓ Phase 1: Evidence Intake (COMPLETE)
- Evidence loading and validation
- SHA-256 hashing
- Metadata extraction
- Session management
- Plate region cropping module

### □ Phase 2: Plate Crop & Upscaling
- Interactive plate region selection
- Upscaling (2x, 4x, 8x)
- Multiple interpolation methods

### □ Phase 3: Image Enhancement
- Grayscale conversion
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Denoising (Bilateral, Non-local means)
- Unsharp masking
- Adaptive thresholding (Gaussian, Mean)

### □ Phase 4: Deblurring
- Wiener deconvolution
- Richardson-Lucy deconvolution
- Motion blur kernel experiments

### □ Phase 5: Video Processing
- Frame extraction
- Quality metrics (Laplacian variance, brightness, contrast)
- Frame sorting and selection

### □ Phase 6: Multi-Frame Analysis
- Frame alignment
- Multi-frame compositing
- Comparative analysis

### □ Phase 7: Comparison Interface
- Side-by-side comparison views
- Method labeling
- Interactive comparison

### □ Phase 8: Manifest & Reporting
- Processing manifest JSON
- HTML forensic report
- Integrity verification section

### □ Phase 9: GUI Implementation
- PySide6 desktop application
- Image viewer with region selection
- Processing pipeline interface
- Results comparison

### □ Phase 10: Testing & Documentation
- Comprehensive unit tests
- Integration tests
- PyInstaller packaging
- User documentation

## Logging

The application uses Python's standard logging with:
- Console output (INFO level)
- Rotating file logs in `logs/` directory
- ISO 8601 timestamps
- Structured log messages

Example log output:
```
2026-09-16 15:30:22 INFO Evidence loaded
2026-09-16 15:30:23 INFO SHA-256 calculated
2026-09-16 15:30:25 INFO Plate crop created
```

## Scientific Limitations

Important forensic considerations:

**Image Enhancement Limitations:**
- Enhancement processes information already present in the source image
- Upscaling, sharpening, and denoising do NOT guarantee recovery of information not captured by original sensor
- Deblurring is experimental and results should be validated
- All enhancement is clearly marked as processed derivative

**AI-Generated Content:**
- If AI-based enhancement is added, it MUST be clearly identified as synthetic/model-generated
- AI outputs must NOT be treated as independent evidence
- Manual verification by investigator is required

**Character Recognition:**
- The application does NOT automatically output recognized plate numbers
- Investigators record observations with confidence levels
- All supporting evidence (original and enhanced frames) is retained

## Testing

### Unit Tests Included

- **test_hashing.py**: SHA-256 calculation and verification
- **test_session.py**: Session creation and management
- **test_evidence.py**: Evidence intake and metadata extraction
- **test_crop.py**: Plate region cropping functionality

### Example Test Run

```bash
$ python -m unittest tests.test_evidence -v
test_evidence_creation_image (tests.test_evidence.TestEvidence) ... ok
test_evidence_image_metadata (tests.test_evidence.TestEvidence) ... ok
test_evidence_sha256_calculated (tests.test_evidence.TestEvidence) ... ok
```

## Future Extensions

This architecture is designed to support future forensic analysis modules:

- Vehicle identification
- Document examination
- Image authentication
- Metadata analysis
- Comparative vehicle tracking

All modules will use the same:
- Evidence management infrastructure
- Integrity verification
- Processing audit trails
- Reporting framework

## License & Legal

This tool is intended for legitimate digital-forensics investigations only.

Users are responsible for:
- Obtaining proper authorization
- Following applicable laws and regulations
- Maintaining chain of custody
- Using evidence appropriately in legal proceedings

## Contributing

Development follows:
- PEP 8 style guidelines
- Type hints throughout
- SOLID principles
- Comprehensive testing
- Clear documentation

## Support

For issues or questions:
1. Check the README and code documentation
2. Review test files for usage examples
3. Check application logs in `logs/` directory

---

**Version**: 1.0.0 (Phase 1)
**Last Updated**: September 2026
