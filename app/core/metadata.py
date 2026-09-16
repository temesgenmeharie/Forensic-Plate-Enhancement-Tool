"""
Metadata management for forensic processing.
Records all operations and maintains audit trail.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ProcessingMetadata:
    """Manages metadata for forensic image processing operations."""
    
    def __init__(self, session_id: str, evidence_id: str, original_sha256: str):
        """
        Initialize processing metadata.
        
        Args:
            session_id: Unique session identifier
            evidence_id: Unique evidence identifier
            original_sha256: SHA-256 hash of original evidence
        """
        self.session_id = session_id
        self.evidence_id = evidence_id
        self.original_sha256 = original_sha256
        self.processing_start_time = datetime.now()
        self.operations: list[dict] = []
        self.derivatives: dict[str, dict] = {}  # Maps filename to hash and metadata
    
    def add_operation(
        self,
        operation_name: str,
        parameters: Optional[dict] = None,
        output_file: Optional[str] = None,
        output_hash: Optional[str] = None
    ) -> None:
        """
        Record an operation performed on the evidence.
        
        Args:
            operation_name: Name of the operation
            parameters: Operation parameters
            output_file: Filename of output derivative
            output_hash: SHA-256 hash of output file
        """
        operation_record = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation_name,
            "parameters": parameters or {},
            "output_file": output_file,
            "output_hash": output_hash
        }
        self.operations.append(operation_record)
        
        if output_file and output_hash:
            self.derivatives[output_file] = {
                "hash": output_hash,
                "operation": operation_name,
                "timestamp": operation_record["timestamp"]
            }
        
        logger.debug(f"Operation recorded: {operation_name}")
    
    def to_dict(self) -> dict:
        """Export metadata as dictionary."""
        return {
            "session_id": self.session_id,
            "evidence_id": self.evidence_id,
            "original_sha256": self.original_sha256,
            "processing_start_time": self.processing_start_time.isoformat(),
            "processing_end_time": datetime.now().isoformat(),
            "software_version": "1.0.0",
            "operations": self.operations,
            "derivatives": self.derivatives
        }
    
    def save_manifest(self, output_path: str | Path) -> None:
        """
        Save processing manifest to JSON file.
        
        Args:
            output_path: Path to save manifest JSON
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        manifest = self.to_dict()
        
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Manifest saved: {output_path}")
    
    @staticmethod
    def load_manifest(manifest_path: str | Path) -> dict:
        """
        Load processing manifest from JSON file.
        
        Args:
            manifest_path: Path to manifest JSON
            
        Returns:
            Manifest dictionary
        """
        manifest_path = Path(manifest_path)
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        logger.info(f"Manifest loaded: {manifest_path}")
        return manifest
