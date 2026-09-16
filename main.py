"""
Forensic License Plate Image & Video Enhancement Tool
Main entry point for Phase 1 & 2 demonstration
"""

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.logging_config import setup_logging
from app.core import Evidence, Session, ProcessingMetadata
from app.processing.pipeline import ProcessingPipeline
from app.processing.crop import PlateRegion
from app.gui_utils import RegionSelector


def demonstrate_phase1():
    """Demonstrate Phase 1 functionality."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("FORENSIC LICENSE PLATE ENHANCEMENT TOOL - PHASE 1")
    logger.info("Evidence Intake, SHA-256 Hashing, and Image Loading")
    logger.info("=" * 70)
    
    print("\n" + "=" * 70)
    print("FORENSIC LICENSE PLATE ENHANCEMENT TOOL - PHASE 1")
    print("Evidence Intake, SHA-256 Hashing, and Image Loading")
    print("=" * 70 + "\n")
    
    # Create a session
    logger.info("Creating analysis session...")
    session = Session()
    print(f"✓ Session created: {session.session_id}")
    logger.info(f"Session: {session.session_id}")
    
    # Check for sample evidence files
    evidence_files = []
    input_dir = Path("input")
    
    if input_dir.exists():
        from app.core import SUPPORTED_FORMATS
        for file_path in input_dir.iterdir():
            if file_path.suffix.lower() in SUPPORTED_FORMATS:
                evidence_files.append(file_path)
    
    if not evidence_files:
        print("\n⚠ No evidence files found in 'input/' directory")
        print("To test with your own files:")
        print("  1. Place image or video files in the 'input/' directory")
        print("  2. Supported formats:")
        print("     - Images: .jpg, .jpeg, .png, .tiff, .bmp")
        print("     - Videos: .mp4, .avi, .mov, .mkv")
        print("  3. Re-run this script\n")
        return session, evidence_files
    
    print(f"✓ Found {len(evidence_files)} evidence file(s)\n")
    
    for evidence_file in evidence_files:
        logger.info(f"Processing evidence: {evidence_file}")
        
        try:
            # Create evidence record
            evidence = Evidence(evidence_file)
            session.add_evidence(evidence.evidence_id)
            
            # Create metadata manager
            metadata = ProcessingMetadata(
                session.session_id,
                evidence.evidence_id,
                evidence.sha256
            )
            
            # Display evidence information
            print(f"\nEvidence: {evidence_file.name}")
            print("-" * 70)
            print(f"  Evidence ID:     {evidence.evidence_id}")
            print(f"  File Type:       {evidence.evidence_type.upper()}")
            print(f"  File Size:       {evidence.file_size:,} bytes")
            print(f"  Resolution:      {evidence.resolution}")
            print(f"  SHA-256:         {evidence.sha256[:32]}...")
            print(f"  Intake Time:     {evidence.intake_timestamp.isoformat()}")
            
            if evidence.evidence_type == "image":
                print(f"  Channels:        {evidence.channels}")
            elif evidence.evidence_type == "video":
                print(f"  FPS:             {evidence.fps:.2f}")
                print(f"  Frame Count:     {evidence.frame_count}")
                print(f"  Duration:        {evidence.duration:.2f}s")
                print(f"  Codec:           {evidence.codec}")
            
            logger.info(f"Evidence loaded successfully: {evidence.evidence_id}")
            
        except Exception as e:
            logger.error(f"Failed to process evidence: {e}")
            print(f"\n✗ Error processing {evidence_file.name}: {e}")
    
    return session, evidence_files


def demonstrate_phase2(session: Session, evidence_files: list[Path]):
    """Demonstrate Phase 2 functionality."""
    logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 70)
    print("FORENSIC LICENSE PLATE ENHANCEMENT TOOL - PHASE 2")
    print("Plate Crop Selection and Upscaling")
    print("=" * 70 + "\n")
    
    # Filter for image files only
    image_files = [f for f in evidence_files if Path(f).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}]
    
    if not image_files:
        print("⚠ No image files available for Phase 2 processing")
        return
    
    for image_file in image_files:
        print(f"\nProcessing: {image_file.name}")
        print("-" * 70)
        
        try:
            # Load image
            image = cv2.imread(str(image_file))
            if image is None:
                logger.error(f"Could not load image: {image_file}")
                continue
            
            # Create evidence record
            evidence = Evidence(image_file)
            session.add_evidence(evidence.evidence_id)
            
            # Initialize processing pipeline
            pipeline = ProcessingPipeline(
                session.session_id,
                evidence.evidence_id,
                evidence.sha256
            )
            
            logger.info(f"Pipeline initialized for {evidence.evidence_id}")
            
            # Save evidence metadata
            pipeline.save_evidence_metadata(evidence.to_dict())
            print(f"✓ Evidence metadata saved")
            
            # Interactive plate region selection (commented for automated demo)
            print("\nSelect plate region:")
            print("  Option 1: Interactive selection (requires display)")
            print("  Option 2: Use default region (100, 100, 400, 150)")
            
            # For demo, use default region
            region = PlateRegion(100, 100, 400, 150)
            print(f"✓ Using region: x={region.x}, y={region.y}, w={region.width}, h={region.height}")
            
            # Validate region
            if region.x + region.width > evidence.width or region.y + region.height > evidence.height:
                print("⚠ Region extends beyond image, adjusting...")
                region = PlateRegion(50, 50, min(300, evidence.width - 100), min(100, evidence.height - 100))
            
            # Process plate crop
            logger.info("Processing plate crop...")
            cropped = pipeline.process_plate_crop(image, region)
            print(f"✓ Plate crop created: {cropped.shape[1]}x{cropped.shape[0]}")
            
            # Process upscaling
            logger.info("Processing upscaling...")
            upscaled = pipeline.process_upscaling(
                cropped,
                scale_factors=[2.0, 4.0, 8.0],
                interpolation="LANCZOS4"
            )
            print(f"✓ Generated {len(upscaled)} upscaled images (2x, 4x, 8x)")
            
            # Save processing manifest
            pipeline.save_processing_manifest()
            print(f"✓ Processing manifest saved")
            
            # Display summary
            summary = pipeline.get_output_summary()
            print(f"\nOutput Summary:")
            print(f"  Output Directory: {summary['output_directory']}")
            print(f"  Files Saved:      {summary['files_saved']}")
            print(f"  Total Operations: {summary['operations_recorded']}")
            
        except Exception as e:
            logger.error(f"Error in Phase 2 processing: {e}")
            print(f"\n✗ Error processing {image_file.name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # Phase 1: Evidence Intake
    session, evidence_files = demonstrate_phase1()
    
    # Display session summary
    print("\n" + "=" * 70)
    print("PHASE 1 SUMMARY")
    print("=" * 70)
    print(f"Session ID:        {session.session_id}")
    print(f"Created At:        {session.created_at.isoformat()}")
    print(f"Evidence Count:    {len(session.evidence_list)}")
    
    # Phase 2: Plate Crop and Upscaling
    if evidence_files:
        demonstrate_phase2(session, evidence_files)
    
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)
    print("\nPhases 1 & 2 implementation includes:")
    print("  ✓ Evidence intake and validation (Phase 1)")
    print("  ✓ SHA-256 hashing for integrity verification (Phase 1)")
    print("  ✓ Image metadata extraction (Phase 1)")
    print("  ✓ Video metadata extraction (Phase 1)")
    print("  ✓ Session management (Phase 1)")
    print("  ✓ Metadata recording (Phase 1)")
    print("  ✓ Plate region cropping module (Phase 1)")
    print("  ✓ Interactive region selection (Phase 2)")
    print("  ✓ Multi-scale upscaling (2x, 4x, 8x) (Phase 2)")
    print("  ✓ Processing pipeline with file management (Phase 2)")
    print("  ✓ Multiple interpolation methods (Phase 2)")
    print("\nNext: Phase 3 - Image enhancement (CLAHE, denoising, sharpening, thresholding)")
    print("=" * 70 + "\n")
