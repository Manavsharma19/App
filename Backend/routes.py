"""
API Routes for HSE Multi-Agent Manager
All endpoint definitions organized using FastAPI's APIRouter
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

# Local imports
from database import get_db
import models

# Import helper functions from main
from main import call_llm, optimize_patient_assignment, LLMConfig

# ============================================
# PYDANTIC MODELS (API Request/Response)
# ============================================
class PatientInput(BaseModel):
    symptoms: str
    patient_name: Optional[str] = None
    age: Optional[int] = None

class TriageResult(BaseModel):
    patient_id: int
    detected_language: str
    translated_symptoms: Optional[str]
    triage_level: int
    triage_reason: str
    specialty_required: str
    chief_complaint: str
    pain_level: Optional[int]
    duration: Optional[str]
    timestamp: str

class Assignment(BaseModel):
    patient_id: int
    hospital_id: int
    hospital_name: str
    reason: str

class HospitalUpdate(BaseModel):
    hospital_id: int
    beds_free: Optional[int] = None
    wait_time: Optional[int] = None

class LLMConfigUpdate(BaseModel):
    provider: str  # "custom", "anthropic", "openai"
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    model_name: Optional[str] = None


# ============================================
# CREATE ROUTER
# ============================================
router = APIRouter(prefix="/api", tags=["api"])


# ============================================
# ENDPOINTS
# ============================================

@router.post("/triage")
async def triage_patient(patient: PatientInput, db: Session = Depends(get_db)):
    """
    Triage a patient using the configured LLM.
    Adds patient to pending queue for optimization.
    """
    # Call LLM for triage
    triage_result = await call_llm(patient.symptoms)
    
    # Get next patient ID
    max_id = db.query(models.Patient.patient_id).order_by(models.Patient.patient_id.desc()).first()
    patient_id = (max_id[0] + 1) if max_id else 100
    
    # Create patient record
    db_patient = models.Patient(
        patient_id=patient_id,
        patient_name=patient.patient_name,
        age=patient.age,
        symptoms=patient.symptoms,
        detected_language=triage_result.get("detected_language", "English"),
        translated_symptoms=triage_result.get("translated_symptoms"),
        triage_level=triage_result.get("triage_level", 4),
        triage_reason=triage_result.get("triage_reason", "Assessment required"),
        specialty_required=triage_result.get("specialty_required", "General"),
        chief_complaint=triage_result.get("chief_complaint", "Not specified"),
        pain_level=triage_result.get("pain_level"),
        duration=triage_result.get("duration"),
        status="pending"
    )
    
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    
    pending_count = db.query(models.Patient).filter(models.Patient.status == "pending").count()
    
    return {
        "success": True,
        "patient": db_patient.to_dict(),
        "pending_count": pending_count,
        "message": f"Patient #{patient_id} triaged as Level {db_patient.triage_level} - {db_patient.specialty_required}"
    }


@router.post("/optimize")
def run_optimization(db: Session = Depends(get_db)):
    """
    Run OR-Tools optimization to assign all pending patients to hospitals.
    """
    result = optimize_patient_assignment(db)
    return result


@router.post("/triage-and-assign")
async def triage_and_assign(patient: PatientInput, db: Session = Depends(get_db)):
    """
    Convenience endpoint: Triage patient AND immediately run optimization.
    Good for real-time single-patient flow.
    """
    # First triage
    triage_response = await triage_patient(patient, db)
    
    # Then optimize
    optimization_result = optimize_patient_assignment(db)
    
    # Find this patient's assignment
    patient_assignment = None
    for assignment in optimization_result.get("assignments", []):
        if assignment["patient_id"] == triage_response["patient"]["patient_id"]:
            patient_assignment = assignment
            break
    
    # Refresh hospitals
    hospitals = db.query(models.Hospital).all()
    
    return {
        "triage": triage_response["patient"],
        "assignment": patient_assignment,
        "optimization_status": optimization_result["status"],
        "hospitals": [h.to_dict() for h in hospitals]
    }


@router.get("/hospitals")
def get_hospitals(db: Session = Depends(get_db)):
    """Get current status of all hospitals"""
    hospitals = db.query(models.Hospital).all()
    
    return {
        "hospitals": [h.to_dict() for h in hospitals],
        "total_beds_free": sum(h.beds_free for h in hospitals),
        "total_beds": sum(h.beds_total for h in hospitals)
    }


@router.patch("/hospitals/{hospital_id}")
def update_hospital(hospital_id: int, update: HospitalUpdate, db: Session = Depends(get_db)):
    """Update hospital capacity or wait time"""
    hospital = db.query(models.Hospital).filter(models.Hospital.id == hospital_id).first()
    
    if not hospital:
        raise HTTPException(status_code=404, detail="Hospital not found")
    
    if update.beds_free is not None:
        hospital.beds_free = update.beds_free
    if update.wait_time is not None:
        hospital.wait_time = update.wait_time
    
    db.commit()
    db.refresh(hospital)
    
    return {"success": True, "hospital": hospital.to_dict()}


@router.get("/patients")
def get_patients(db: Session = Depends(get_db)):
    """Get all patient queues"""
    pending = db.query(models.Patient).filter(models.Patient.status == "pending").all()
    assigned = db.query(models.Patient).filter(models.Patient.status == "assigned").order_by(models.Patient.assigned_at.desc()).limit(20).all()
    
    return {
        "pending": [p.to_dict() for p in pending],
        "assigned": [p.to_dict() for p in assigned],
        "pending_count": len(pending),
        "assigned_count": db.query(models.Patient).filter(models.Patient.status == "assigned").count()
    }


@router.get("/activity")
def get_activity(db: Session = Depends(get_db)):
    """Get recent activity log"""
    activities = db.query(models.Activity).order_by(models.Activity.timestamp.desc()).limit(20).all()
    
    return {
        "activity": [a.to_dict() for a in activities]
    }


@router.post("/config/llm")
def update_llm_config(config: LLMConfigUpdate):
    """
    Update LLM configuration at runtime.
    Use this to point to your fine-tuned model.
    """
    LLMConfig.PROVIDER = config.provider
    
    if config.endpoint:
        LLMConfig.CUSTOM_ENDPOINT = config.endpoint
    if config.api_key:
        LLMConfig.CUSTOM_API_KEY = config.api_key
    if config.model_name:
        LLMConfig.CUSTOM_MODEL_NAME = config.model_name
    
    return {
        "success": True,
        "config": {
            "provider": LLMConfig.PROVIDER,
            "endpoint": LLMConfig.CUSTOM_ENDPOINT if config.provider == "custom" else None,
            "model": LLMConfig.CUSTOM_MODEL_NAME if config.provider == "custom" else None
        }
    }


@router.get("/config/llm")
def get_llm_config():
    """Get current LLM configuration"""
    return {
        "provider": LLMConfig.PROVIDER,
        "custom_endpoint": LLMConfig.CUSTOM_ENDPOINT,
        "custom_model": LLMConfig.CUSTOM_MODEL_NAME,
        "has_anthropic_key": bool(LLMConfig.ANTHROPIC_API_KEY),
        "has_openai_key": bool(LLMConfig.OPENAI_API_KEY)
    }


@router.post("/reset")
def reset_all(db: Session = Depends(get_db)):
    """Reset all data for demo"""
    # Delete all patients and activities
    db.query(models.Patient).delete()
    db.query(models.Activity).delete()
    
    # Reset hospital beds to default values
    hospitals = db.query(models.Hospital).all()
    default_beds = {0: 12, 1: 18, 2: 8}
    
    for hospital in hospitals:
        if hospital.id in default_beds:
            hospital.beds_free = default_beds[hospital.id]
    
    db.commit()
    
    return {"success": True, "message": "All data reset"}
