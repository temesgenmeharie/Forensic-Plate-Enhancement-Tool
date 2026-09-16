"""
Processing manifest schema and generation module.
Defines the structure for forensic processing manifests in JSON format.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ProcessingManifestSchema:
    """Schema definition for processing manifest."""
    
    @staticmethod
    def create_manifest(
        session_id: str,
        evidence_id: str,
        original_sha256: str,
        operations: List[Dict],
        software_version: str = "1.0.0"
    ) -> Dict:
        """
        Create processing manifest dictionary.
        
        Args:
            session_id: Analysis session ID
            evidence_id: Evidence ID
            original_sha256: SHA-256 of original evidence
            operations: List of operation dictionaries
            software_version: Software version (default: 1.0.0)
            
        Returns:
            Manifest dictionary
        """
        manifest = {
            "forensic_manifest": {
                "schema_version": "1.0",
                "manifest_date": datetime.now().isoformat(),
                "software": {
                    "name": "Forensic Plate Enhancer",
                    "version": software_version
                },
                "session": {
                    "session_id": session_id,
                    "start_date": datetime.now().isoformat()
                },
                "evidence": {
                    "evidence_id": evidence_id,
                    "original_sha256": original_sha256,
                    "processing_start": datetime.now().isoformat()
                },
                "processing": {
                    "operations_count": len(operations),
                    "operations": operations
                },
                "integrity": {
                    "original_preserved": True,
                    "audit_trail_maintained": True
                }
            }
        }
        
        logger.debug("Manifest created")
        
        return manifest
    
    @staticmethod
    def save_manifest(
        manifest: Dict,
        output_path: str | Path
    ) -> Path:
        """
        Save manifest to JSON file.
        
        Args:
            manifest: Manifest dictionary
            output_path: Path to save manifest
            
        Returns:
            Path to saved manifest
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Manifest saved: {output_path}")
        
        return output_path
    
    @staticmethod
    def load_manifest(manifest_path: str | Path) -> Dict:
        """
        Load manifest from JSON file.
        
        Args:
            manifest_path: Path to manifest file
            
        Returns:
            Manifest dictionary
        """
        manifest_path = Path(manifest_path)
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        logger.info(f"Manifest loaded: {manifest_path}")
        
        return manifest
    
    @staticmethod
    def validate_manifest(manifest: Dict) -> bool:
        """
        Validate manifest structure.
        
        Args:
            manifest: Manifest dictionary
            
        Returns:
            True if manifest is valid
        """
        required_keys = ["forensic_manifest"]
        
        for key in required_keys:
            if key not in manifest:
                logger.error(f"Missing required key in manifest: {key}")
                return False
        
        manifest_data = manifest["forensic_manifest"]
        
        required_manifest_keys = [
            "schema_version",
            "software",
            "session",
            "evidence",
            "processing"
        ]
        
        for key in required_manifest_keys:
            if key not in manifest_data:
                logger.error(f"Missing required key in forensic_manifest: {key}")
                return False
        
        logger.debug("Manifest validation passed")
        
        return True


class ManifestGenerator:
    """High-level manifest generation utility."""
    
    def __init__(
        self,
        session_id: str,
        evidence_id: str,
        original_sha256: str,
        software_version: str = "1.0.0"
    ):
        """
        Initialize manifest generator.
        
        Args:
            session_id: Analysis session ID
            evidence_id: Evidence ID
            original_sha256: SHA-256 of original evidence
            software_version: Software version (default: 1.0.0)
        """
        self.session_id = session_id
        self.evidence_id = evidence_id
        self.original_sha256 = original_sha256
        self.software_version = software_version
        self.operations: List[Dict] = []
        self.derivatives: Dict[str, str] = {}
    
    def add_operation(
        self,
        operation_name: str,
        parameters: Optional[Dict] = None,
        output_file: Optional[str] = None,
        output_hash: Optional[str] = None
    ) -> None:
        """
        Add operation to manifest.
        
        Args:
            operation_name: Name of operation
            parameters: Operation parameters
            output_file: Output file name
            output_hash: SHA-256 of output file
        """
        operation = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation_name,
            "parameters": parameters or {},
            "output": {
                "filename": output_file,
                "sha256": output_hash
            }
        }
        
        self.operations.append(operation)
        
        if output_file and output_hash:
            self.derivatives[output_file] = output_hash
        
        logger.debug(f"Operation added: {operation_name}")
    
    def generate_manifest(self) -> Dict:
        """
        Generate complete manifest.
        
        Returns:
            Manifest dictionary
        """
        return ProcessingManifestSchema.create_manifest(
            self.session_id,
            self.evidence_id,
            self.original_sha256,
            self.operations,
            self.software_version
        )
    
    def save_manifest(self, output_path: str | Path) -> Path:
        """
        Save manifest to file.
        
        Args:
            output_path: Path to save manifest
            
        Returns:
            Path to saved manifest
        """
        manifest = self.generate_manifest()
        return ProcessingManifestSchema.save_manifest(manifest, output_path)
    
    def get_operations_count(self) -> int:
        """Get number of operations."""
        return len(self.operations)
    
    def get_derivatives_count(self) -> int:
        """Get number of derivative files."""
        return len(self.derivatives)
    
    def get_operations_summary(self) -> str:
        """Get text summary of operations."""
        summary = f"Processing Summary:\n"
        summary += f"Session: {self.session_id}\n"
        summary += f"Evidence: {self.evidence_id}\n"
        summary += f"Operations: {len(self.operations)}\n"
        summary += f"Derivatives: {len(self.derivatives)}\n"
        
        return summary
