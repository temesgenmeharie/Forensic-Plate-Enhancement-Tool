"""Tests for video reader module."""

import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from app.video.reader import VideoReader


class TestVideoReader(unittest.TestCase):
    """Test video reading functionality."""
    
    def setUp(self):
        """Create a test video file."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        # Create a simple test video
        self.video_path = self.temp_path / "test_video.mp4"
        self._create_test_video(self.video_path)
    
    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()
    
    def _create_test_video(self, output_path: Path, frames: int = 30, fps: int = 30):
        """Create a simple test video."""
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Create test frames (simple solid colors)
        for i in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Add some variation (gradient)
            frame[:, :] = [i * 8 % 256, i * 4 % 256, i * 2 % 256]
            writer.write(frame)
        
        writer.release()
    
    def test_video_reader_initialization(self):
        """Test video reader initialization."""
        reader = VideoReader(self.video_path)
        
        self.assertEqual(reader.width, 640)
        self.assertEqual(reader.height, 480)
        self.assertEqual(reader.fps, 30)
        self.assertEqual(reader.frame_count, 30)
        self.assertAlmostEqual(reader.duration, 1.0, places=1)
        
        reader.close()
    
    def test_video_reader_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with self.assertRaises(FileNotFoundError):
            VideoReader(self.temp_path / "nonexistent.mp4")
    
    def test_get_metadata(self):
        """Test getting video metadata."""
        reader = VideoReader(self.video_path)
        
        metadata = reader.get_metadata()
        
        self.assertIn("filename", metadata)
        self.assertIn("width", metadata)
        self.assertIn("height", metadata)
        self.assertIn("fps", metadata)
        self.assertIn("frame_count", metadata)
        self.assertEqual(metadata["width"], 640)
        self.assertEqual(metadata["height"], 480)
        
        reader.close()
    
    def test_read_frame(self):
        """Test reading individual frames."""
        reader = VideoReader(self.video_path)
        
        success, frame = reader.read_frame(0)
        
        self.assertTrue(success)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        
        reader.close()
    
    def test_read_frame_out_of_range(self):
        """Test reading frame beyond video range."""
        reader = VideoReader(self.video_path)
        
        success, frame = reader.read_frame(1000)
        
        self.assertFalse(success)
        self.assertIsNone(frame)
        
        reader.close()
    
    def test_read_frame_range(self):
        """Test reading frame range."""
        reader = VideoReader(self.video_path)
        
        frames = reader.read_frame_range(start_frame=0, end_frame=5, step=1)
        
        self.assertEqual(len(frames), 6)  # Frames 0-5 inclusive
        
        reader.close()
    
    def test_read_frame_range_with_step(self):
        """Test reading frame range with step."""
        reader = VideoReader(self.video_path)
        
        frames = reader.read_frame_range(start_frame=0, end_frame=10, step=2)
        
        self.assertEqual(len(frames), 6)  # Frames 0, 2, 4, 6, 8, 10
        
        reader.close()
    
    def test_read_time_range(self):
        """Test reading frames by time range."""
        reader = VideoReader(self.video_path)
        
        frames = reader.read_time_range(start_time=0.0, end_time=0.5, interval=0.1)
        
        self.assertGreater(len(frames), 0)
        
        reader.close()
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with VideoReader(self.video_path) as reader:
            self.assertEqual(reader.frame_count, 30)


if __name__ == "__main__":
    unittest.main()
