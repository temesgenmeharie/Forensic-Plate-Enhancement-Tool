"""Tests for hashing module."""

import tempfile
from pathlib import Path
import unittest

from app.core.hashing import calculate_sha256, calculate_data_sha256, verify_hash


class TestHashing(unittest.TestCase):
    """Test SHA-256 hashing functionality."""
    
    def setUp(self):
        """Create temporary test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_calculate_sha256_file(self):
        """Test SHA-256 calculation on a file."""
        test_file = self.temp_path / "test.txt"
        test_data = b"forensic evidence test data"
        test_file.write_bytes(test_data)
        
        hash_value = calculate_sha256(test_file)
        
        # Verify it's a valid hex string of correct length
        self.assertEqual(len(hash_value), 64)
        self.assertTrue(all(c in '0123456789abcdef' for c in hash_value))
    
    def test_calculate_sha256_consistency(self):
        """Test that SHA-256 calculation is consistent."""
        test_file = self.temp_path / "test.txt"
        test_data = b"forensic evidence test data"
        test_file.write_bytes(test_data)
        
        hash1 = calculate_sha256(test_file)
        hash2 = calculate_sha256(test_file)
        
        self.assertEqual(hash1, hash2)
    
    def test_calculate_sha256_different_files(self):
        """Test that different files have different hashes."""
        file1 = self.temp_path / "test1.txt"
        file2 = self.temp_path / "test2.txt"
        
        file1.write_bytes(b"data1")
        file2.write_bytes(b"data2")
        
        hash1 = calculate_sha256(file1)
        hash2 = calculate_sha256(file2)
        
        self.assertNotEqual(hash1, hash2)
    
    def test_calculate_sha256_file_not_found(self):
        """Test that FileNotFoundError is raised for missing files."""
        with self.assertRaises(FileNotFoundError):
            calculate_sha256(self.temp_path / "nonexistent.txt")
    
    def test_calculate_data_sha256(self):
        """Test SHA-256 calculation on binary data."""
        data = b"forensic evidence test data"
        hash_value = calculate_data_sha256(data)
        
        self.assertEqual(len(hash_value), 64)
        self.assertTrue(all(c in '0123456789abcdef' for c in hash_value))
    
    def test_verify_hash_match(self):
        """Test hash verification with matching hash."""
        test_file = self.temp_path / "test.txt"
        test_data = b"forensic evidence test data"
        test_file.write_bytes(test_data)
        
        hash_value = calculate_sha256(test_file)
        is_valid = verify_hash(test_file, hash_value)
        
        self.assertTrue(is_valid)
    
    def test_verify_hash_mismatch(self):
        """Test hash verification with mismatched hash."""
        test_file = self.temp_path / "test.txt"
        test_file.write_bytes(b"test data")
        
        wrong_hash = "0" * 64
        is_valid = verify_hash(test_file, wrong_hash)
        
        self.assertFalse(is_valid)
    
    def test_verify_hash_case_insensitive(self):
        """Test that hash verification is case-insensitive."""
        test_file = self.temp_path / "test.txt"
        test_file.write_bytes(b"test data")
        
        hash_value = calculate_sha256(test_file)
        is_valid = verify_hash(test_file, hash_value.upper())
        
        self.assertTrue(is_valid)


if __name__ == "__main__":
    unittest.main()
