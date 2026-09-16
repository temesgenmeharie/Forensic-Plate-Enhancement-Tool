"""Tests for frame extractor module."""

import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from app.video.frame_extractor import FrameExtractor


class TestFrameExtractor(unittest.TestCase):
    """Test frame extraction functionality."""
    
    def setUp(self):
        """Create a test video file."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create output directory for frames
        self.output_dir = self.temp_path / "frames"
        self.output_dir.mkdir()
        
        # Create a test video
        self.video_path = self.temp_path / "test_video.mp4"
        self._create_test_video(self.video_path, frames=30)
    
    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()
    
    def _create_test_video(self, output_path: Path, frames: int = 30, fps: int = 30):
        """Create a simple test video."""
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        for i in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = [i * 8 % 256, i * 4 % 256, i * 2 % 256]
            writer.write(frame)
        
        writer.release()
    
    def test_frame_extractor_initialization(self):
        """Test frame extractor initialization."""
        extractor = FrameExtractor(self.video_path, output_dir=self.output_dir)
        
        self.assertIsNotNone(extractor.video_reader)
        self.assertEqual(extractor.output_dir, self.output_dir)
        
        extractor.close()
    
    def test_extract_keyframes(self):
        """Test keyframe extraction."""
        extractor = FrameExtractor(self.video_path)
        
        frames = extractor.extract_keyframes(interval=0.1)
        
        self.assertGreater(len(frames), 0)
        
        # Check structure
        for frame_num, frame, metrics in frames:
            self.assertIsInstance(frame_num, int)
            self.assertIsNotNone(frame)
            self.assertIsInstance(metrics, dict)
        
        extractor.close()
    
    def test_extract_all_frames(self):
        """Test extracting all frames."""
        extractor = FrameExtractor(self.video_path)
        
        frames = extractor.extract_all_frames(stride=2)
        
        self.assertGreater(len(frames), 0)
        
        extractor.close()
    
    def test_extract_frame_range(self):
        """Test extracting specific frame range."""
        extractor = FrameExtractor(self.video_path)
        
        frames = extractor.extract_all_frames(start_frame=0, end_frame=10)
        
        self.assertEqual(len(frames), 11)
        
        extractor.close()
    
    def test_select_best_frames_sharpness(self):
        """Test selecting best frames by sharpness."""
        extractor = FrameExtractor(self.video_path)
        
        frames = extractor.extract_keyframes(interval=0.1)
        selected = extractor.select_best_frames(
            frames,
            selection_method="sharpness",
            count=3
        )
        
        self.assertEqual(len(selected), 3)
        
        extractor.close()
    
    def test_select_best_frames_composite(self):
        """Test selecting best frames by composite score."""
        extractor = FrameExtractor(self.video_path)
        
        frames = extractor.extract_keyframes(interval=0.1)
        selected = extractor.select_best_frames(
            frames,
            selection_method="composite",
            count=3
        )
        
        self.assertEqual(len(selected), 3)
        
        extractor.close()
    
    def test_save_frames(self):
        """Test saving extracted frames."""
        extractor = FrameExtractor(self.video_path, output_dir=self.output_dir)
        
        frames = extractor.extract_keyframes(interval=0.5)
        selected = extractor.select_best_frames(frames, count=2)
        
        saved_paths = extractor.save_frames(selected, prefix="test_frame")
        
        self.assertEqual(len(saved_paths), 2)
        
        # Verify files exist
        for path in saved_paths:
            self.assertTrue(path.exists())
        
        extractor.close()
    
    def test_get_video_metadata(self):
        """Test getting video metadata."""
        extractor = FrameExtractor(self.video_path)
        
        metadata = extractor.get_video_metadata()
        
        self.assertIn("width", metadata)
        self.assertIn("height", metadata)
        self.assertIn("fps", metadata)
        self.assertIn("frame_count", metadata)
        
        extractor.close()
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with FrameExtractor(self.video_path) as extractor:
            metadata = extractor.get_video_metadata()
            self.assertIsNotNone(metadata)


if __name__ == "__main__":
    unittest.main()
