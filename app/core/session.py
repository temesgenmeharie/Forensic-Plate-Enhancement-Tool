"""
Session management for forensic analysis workflows.
Generates unique session IDs and manages session metadata.
"""

import uuid
from datetime import datetime
from typing import Optional


class Session:
    """Represents a forensic analysis session."""
    
    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize a forensic analysis session.
        
        Args:
            session_id: Optional existing session ID. If not provided, generates new one.
        """
        self.session_id = session_id or self._generate_session_id()
        self.created_at = datetime.now()
        self.evidence_list: list[str] = []
        self.operations: list[dict] = []
    
    @staticmethod
    def _generate_session_id() -> str:
        """
        Generate a unique session ID.
        
        Format: SES-YYYY-XXXXXX where XXXXXX is random hex digits
        """
        timestamp = datetime.now().strftime("%Y")
        random_part = uuid.uuid4().hex[:6].upper()
        return f"SES-{timestamp}-{random_part}"
    
    def add_evidence(self, evidence_id: str) -> None:
        """Add evidence to this session."""
        if evidence_id not in self.evidence_list:
            self.evidence_list.append(evidence_id)
    
    def add_operation(self, operation_name: str, **kwargs) -> None:
        """
        Record an operation performed in this session.
        
        Args:
            operation_name: Name of the operation
            **kwargs: Operation parameters
        """
        operation = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation_name,
            **kwargs
        }
        self.operations.append(operation)
    
    def to_dict(self) -> dict:
        """Export session metadata as dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "evidence_count": len(self.evidence_list),
            "operation_count": len(self.operations)
        }
