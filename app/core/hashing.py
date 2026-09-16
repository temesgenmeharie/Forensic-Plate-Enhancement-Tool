"""
SHA-256 hashing and integrity verification for forensic evidence.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def calculate_sha256(file_path: str | Path, chunk_size: int = 8192) -> str:
    """
    Calculate SHA-256 hash of a file.
    
    Args:
        file_path: Path to file
        chunk_size: Size of chunks to read (default 8192 bytes)
        
    Returns:
        Hexadecimal SHA-256 hash string
        
    Raises:
        FileNotFoundError: If file does not exist
        IOError: If file cannot be read
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    sha256_hash = hashlib.sha256()
    
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256_hash.update(chunk)
        
        hash_value = sha256_hash.hexdigest()
        logger.info(f"SHA-256 calculated for {file_path.name}: {hash_value[:16]}...")
        return hash_value
        
    except IOError as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        raise


def calculate_data_sha256(data: bytes) -> str:
    """
    Calculate SHA-256 hash of binary data.
    
    Args:
        data: Binary data
        
    Returns:
        Hexadecimal SHA-256 hash string
    """
    sha256_hash = hashlib.sha256(data).hexdigest()
    logger.debug(f"SHA-256 calculated for data: {sha256_hash[:16]}...")
    return sha256_hash


def verify_hash(file_path: str | Path, expected_hash: str) -> bool:
    """
    Verify that a file's SHA-256 hash matches the expected value.
    
    Args:
        file_path: Path to file
        expected_hash: Expected SHA-256 hash value
        
    Returns:
        True if hash matches, False otherwise
    """
    calculated_hash = calculate_sha256(file_path)
    matches = calculated_hash.lower() == expected_hash.lower()
    
    if matches:
        logger.info(f"Hash verification successful for {file_path}")
    else:
        logger.warning(f"Hash mismatch for {file_path}")
    
    return matches
