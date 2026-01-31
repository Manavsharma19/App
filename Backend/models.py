"""
SQLAlchemy database models for HSE Multi-Agent Manager
"""
from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, Text
from database import Base
from datetime import datetime

class Hospital(Base):
    """Hospital capacity and information"""
    __tablename__ = "hospitals"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    beds_total = Column(Integer)
    beds_free = Column(Integer)
    wait_time = Column(Integer)  # in minutes
    specialties = Column(JSON)  # List of specialties
    location = Column(JSON)  # {lat, lng}
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "name": self.name,
            "beds_total": self.beds_total,
            "beds_free": self.beds_free,
            "wait_time": self.wait_time,
            "specialties": self.specialties,
            "location": self.location
        }


class Patient(Base):
    """Patient triage and assignment records"""
    __tablename__ = "patients"
    
    patient_id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    symptoms = Column(Text)
    detected_language = Column(String)
    translated_symptoms = Column(Text, nullable=True)
    triage_level = Column(Integer)  # 1-5
    triage_reason = Column(Text)
    specialty_required = Column(String)
    chief_complaint = Column(String)
    pain_level = Column(Integer, nullable=True)  # 1-10
    duration = Column(String, nullable=True)
    status = Column(String)  # "pending" or "assigned"
    assigned_hospital = Column(String, nullable=True)
    assigned_at = Column(DateTime, nullable=True)  # ← THIS WAS MISSING
    timestamp = Column(DateTime, default=datetime.now)
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "age": self.age,
            "symptoms": self.symptoms,
            "detected_language": self.detected_language,
            "translated_symptoms": self.translated_symptoms,
            "triage_level": self.triage_level,
            "triage_reason": self.triage_reason,
            "specialty_required": self.specialty_required,
            "chief_complaint": self.chief_complaint,
            "pain_level": self.pain_level,
            "duration": self.duration,
            "status": self.status,
            "assigned_hospital": self.assigned_hospital,
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "timestamp": self.timestamp.isoformat()
        }


class Activity(Base):
    """Activity log for dashboard"""
    __tablename__ = "activity_log"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(String)  # HH:MM format
    patient_id = Column(Integer)
    hospital = Column(String)
    specialty = Column(String)
    urgency = Column(Integer)  # 1-5 triage level
    timestamp = Column(DateTime, default=datetime.now)  # ← THIS WAS MISSING
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "time": self.time,
            "patient_id": self.patient_id,
            "hospital": self.hospital,
            "specialty": self.specialty,
            "urgency": self.urgency
        }
