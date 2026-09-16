"""
Forensic License Plate Image & Video Enhancement Tool
Main entry point for Phase 1 demonstration
"""

import logging
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.logging_config import setup_logging
from app.core import Evidence, Session, ProcessingMetadata


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
    else:
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
    
    # Display session summary
    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    print(f"Session ID:        {session.session_id}")
    print(f"Created At:        {session.created_at.isoformat()}")
    print(f"Evidence Count:    {len(session.evidence_list)}")
    print(f"Operations:        {len(session.operations)}")
    
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print("=" * 70)
    print("\nPhase 1 implementation includes:")
    print("  ✓ Evidence intake and validation")
    print("  ✓ SHA-256 hashing for integrity verification")
    print("  ✓ Image metadata extraction")
    print("  ✓ Video metadata extraction")
    print("  ✓ Session management")
    print("  ✓ Metadata recording")
    print("  ✓ Plate region cropping module")
    print("\nNext: Phase 2 - Plate crop selection and upscaling")
    print("=" * 70 + "\n")
    
    logger.info("Phase 1 demonstration complete")


if __name__ == "__main__":
    demonstrate_phase1()
