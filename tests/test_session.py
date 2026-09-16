"""Tests for session management module."""

import unittest
from app.core.session import Session


class TestSession(unittest.TestCase):
    """Test Session functionality."""
    
    def test_session_creation(self):
        """Test that a session can be created."""
        session = Session()
        
        self.assertIsNotNone(session.session_id)
        self.assertTrue(session.session_id.startswith("SES-"))
        self.assertIsNotNone(session.created_at)
    
    def test_session_id_format(self):
        """Test that session ID has correct format."""
        session = Session()
        parts = session.session_id.split("-")
        
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0], "SES")
        self.assertEqual(len(parts[1]), 4)  # Year
        self.assertEqual(len(parts[2]), 6)  # Random hex
    
    def test_session_id_uniqueness(self):
        """Test that different sessions have different IDs."""
        session1 = Session()
        session2 = Session()
        
        # Very unlikely to be equal due to random component
        # (unless they're created in exact same microsecond)
        self.assertNotEqual(session1.session_id, session2.session_id)
    
    def test_custom_session_id(self):
        """Test creating a session with a custom ID."""
        custom_id = "SES-2026-CUSTOM"
        session = Session(session_id=custom_id)
        
        self.assertEqual(session.session_id, custom_id)
    
    def test_add_evidence(self):
        """Test adding evidence to session."""
        session = Session()
        
        session.add_evidence("EV-001")
        session.add_evidence("EV-002")
        
        self.assertEqual(len(session.evidence_list), 2)
        self.assertIn("EV-001", session.evidence_list)
    
    def test_add_evidence_no_duplicates(self):
        """Test that evidence is not added twice."""
        session = Session()
        
        session.add_evidence("EV-001")
        session.add_evidence("EV-001")
        
        self.assertEqual(len(session.evidence_list), 1)
    
    def test_add_operation(self):
        """Test recording operations."""
        session = Session()
        
        session.add_operation("crop", x=0, y=0, width=100, height=50)
        
        self.assertEqual(len(session.operations), 1)
        self.assertEqual(session.operations[0]["operation"], "crop")
    
    def test_to_dict(self):
        """Test exporting session as dictionary."""
        session = Session()
        session.add_evidence("EV-001")
        session.add_operation("crop", x=0, y=0)
        
        data = session.to_dict()
        
        self.assertIn("session_id", data)
        self.assertIn("created_at", data)
        self.assertEqual(data["evidence_count"], 1)
        self.assertEqual(data["operation_count"], 1)


if __name__ == "__main__":
    unittest.main()
