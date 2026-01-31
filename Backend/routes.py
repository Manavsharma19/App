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

TRIAGE_SYSTEM_PROMPT = """You are an HSE (Health Service Executive, Ireland) triage nurse assistant.
Your job is to assess patient symptoms and provide triage information.

Given patient symptoms, you MUST return ONLY valid JSON with no additional text, markdown, or explanation:

{
  "detected_language": "Irish" or "English",
  "translated_symptoms": "English translation if input was Irish, otherwise null",
  "triage_level": 1-5,
  "triage_reason": "Brief clinical explanation for the triage level",
  "specialty_required": "One of: Cardiology, Neurology, General, Orthopaedics, Trauma, Maternity, Oncology",
  "chief_complaint": "Main issue in 3-5 words",
  "pain_level": 1-10 or null if not mentioned,
  "duration": "How long symptoms have been present, or null"
}

Triage Levels (Manchester Triage System):
- 1 = Immediate (life-threatening, e.g., cardiac arrest, severe breathing difficulty)
- 2 = Very Urgent (e.g., chest pain, severe bleeding, stroke symptoms)
- 3 = Urgent (e.g., moderate pain, fractures, high fever)
- 4 = Standard (e.g., minor injuries, mild symptoms)
- 5 = Non-Urgent (e.g., minor complaints, routine issues)

IMPORTANT: Return ONLY the JSON object, no other text."""

@router.post("/triage")
async def triage_patient(patient: PatientInput, db: Session = Depends(get_db)):
    """
    Triage a patient using the configured LLM.
    Adds patient to pending queue for optimization.
    """
    # Call LLM for triage
    triage_result = await call_llm(patient.symptoms, TRIAGE_SYSTEM_PROMPT)
    
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


# ============================================
# Dischatge Logic
# ============================================

# ============================================
# DISCHARGE DATA MODELS
# ============================================
class DischargeAssessment(BaseModel):
    patient_id: int
    hospital_id: int
    final_verdict: str


# ============================================
# LLM PROMPTS FOR DISCHARGE
# ============================================
DISCHARGE_FORM_PROMPT = """You are an HSE discharge coordinator. Given patient data, generate a structured discharge assessment form.

Return ONLY valid JSON:
{
    "patient_summary": "2-3 sentence summary of patient's presentation",
    "key_findings": ["finding1", "finding2", "finding3"],
    "form_fields": [
        {"name": "clinical_outcome", "label": "Clinical Outcome", "type": "textarea", "placeholder": "How did the patient respond to treatment?"},
        {"name": "current_status", "label": "Current Status", "type": "select", "options": ["Improved", "Stable", "Worsened"]},
        {"name": "follow_up_needed", "label": "Follow-up Required?", "type": "checkbox"},
        {"name": "additional_notes", "label": "Additional Notes", "type": "textarea"}
    ],
    "suggested_questions": ["Was patient compliant with treatment?", "Any adverse reactions?", "Ready for discharge?"]
}"""


REFERRAL_LETTER_PROMPT = """You are an HSE discharge coordinator drafting a GP referral letter.

Given patient data and nurse's final verdict, generate a professional referral letter.

Return ONLY valid JSON:
{
    "letter_date": "today's date",
    "gp_salutation": "Dear Dr. [GP Name]",
    "patient_info": "Patient name, age, ID",
    "reason_for_presentation": "Why patient came to hospital",
    "treatment_provided": "What was done in hospital",
    "current_status": "Patient's status at discharge",
    "recommendations": ["recommendation1", "recommendation2"],
    "follow_up_timeline": "When patient should see GP",
    "letter_body": "Full professional letter text",
    "signature": "From: HSE [Hospital Name]"
}"""


# ============================================
# DISCHARGE HELPER FUNCTIONS
# ============================================
async def generate_discharge_form(patient_data: dict) -> dict:
    """Use LLM to generate a discharge assessment form based on patient data."""
    
    patient_summary = f"""
    Patient: {patient_data.get('patient_name', 'Unknown')} (Age: {patient_data.get('age', '?')})
    Presented with: {patient_data.get('symptoms', 'Not specified')}
    Triage Level: {patient_data.get('triage_level', 4)}
    Specialty: {patient_data.get('specialty_required', 'General')}
    Hospital: {patient_data.get('assigned_hospital', 'Unknown')}
    """
    
    # Reuse the call_llm pattern from main.py
    try:
        result = await call_llm(patient_summary, system_prompt=DISCHARGE_FORM_PROMPT)
        return result
    except:
        return mock_discharge_form()


def mock_discharge_form() -> dict:
    """Fallback discharge form generator."""
    return {
        "patient_summary": "Patient presented with acute symptoms and received appropriate treatment.",
        "key_findings": ["Condition stabilized", "Vital signs within normal range", "Responding well to treatment"],
        "form_fields": [
            {"name": "clinical_outcome", "label": "Clinical Outcome", "type": "textarea", "placeholder": "How did the patient respond to treatment?"},
            {"name": "current_status", "label": "Current Status", "type": "select", "options": ["Improved", "Stable", "Worsened"]},
            {"name": "follow_up_needed", "label": "Follow-up Required?", "type": "checkbox"},
            {"name": "medications", "label": "Discharge Medications", "type": "textarea"}
        ],
        "suggested_questions": ["Was patient compliant?", "Any complications?", "Ready for discharge?"]
    }


async def generate_referral_letter(patient_data: dict, nurse_verdict: str) -> dict:
    """Use LLM to generate a GP referral letter."""
    
    context = f"""
    Patient: {patient_data.get('patient_name')}
    Age: {patient_data.get('age')}
    Presenting Complaint: {patient_data.get('symptoms')}
    Hospital: {patient_data.get('assigned_hospital', 'HSE Hospital')}
    
    Nurse's Final Verdict:
    {nurse_verdict}
    """
    
    try:
        result = await call_llm(context, system_prompt=REFERRAL_LETTER_PROMPT)
        return result
    except:
        return mock_referral_letter(patient_data, nurse_verdict)


def mock_referral_letter(patient_data: dict, nurse_verdict: str) -> dict:
    """Fallback referral letter generator."""
    return {
        "letter_date": datetime.now().strftime("%d/%m/%Y"),
        "gp_salutation": "Dear General Practitioner",
        "patient_info": f"{patient_data.get('patient_name')}, Age {patient_data.get('age')}",
        "reason_for_presentation": patient_data.get('symptoms', 'Not specified'),
        "treatment_provided": "Patient received appropriate care and monitoring during hospital stay",
        "current_status": "Patient discharged in stable condition",
        "recommendations": ["Monitor vital signs", "Ensure medication compliance", "Return if symptoms worsen"],
        "follow_up_timeline": "Within 1-2 weeks",
        "letter_body": f"Dear GP,\n\nRe: {patient_data.get('patient_name')}\n\nThis patient was admitted with {patient_data.get('symptoms')}.\n\nNurse Assessment: {nurse_verdict}\n\nPlease arrange follow-up within 1-2 weeks.\n\nBest regards,\nHSE Discharge Team",
        "signature": f"HSE {patient_data.get('assigned_hospital', 'Hospital')}"
    }


# ============================================
# DISCHARGE ENDPOINTS
# ============================================
@router.post("/discharge/form")
async def get_discharge_form(patient_id: int, db: Session = Depends(get_db)):
    """Get discharge assessment form for a patient."""
    
    patient = db.query(models.Patient).filter(models.Patient.patient_id == patient_id).first()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    patient_data = patient.to_dict()
    form = await generate_discharge_form(patient_data)
    
    return {
        "success": True,
        "patient_id": patient_id,
        "patient_name": patient.patient_name,
        "form": form
    }


@router.post("/discharge/submit")
async def submit_discharge(assessment: DischargeAssessment, db: Session = Depends(get_db)):
    """Submit discharge assessment. If not cured, generate referral letter."""
    
    patient = db.query(models.Patient).filter(models.Patient.patient_id == assessment.patient_id).first()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get hospital
    hospital = db.query(models.Hospital).filter(models.Hospital.id == assessment.hospital_id).first()
    
    # If cured, just discharge
    if assessment.final_verdict.lower() == "cured":
        patient.status = "discharged"
        patient.discharged_at = datetime.now()
        
        # Free up bed
        if hospital:
            hospital.beds_free += 1
        
        # Log activity
        activity = models.Activity(
            time=datetime.now().strftime("%H:%M"),
            patient_id=patient.patient_id,
            hospital=patient.assigned_hospital,
            specialty=patient.specialty_required,
            urgency=patient.triage_level,
            action="DISCHARGED - CURED"
        )
        db.add(activity)
        db.commit()
        
        return {
            "success": True,
            "status": "discharged_cured",
            "message": f"Patient #{patient.patient_id} successfully discharged"
        }
    
    # Generate referral letter
    patient_data = patient.to_dict()
    referral = await generate_referral_letter(patient_data, assessment.final_verdict)
    
    patient.status = "discharged"
    patient.discharged_at = datetime.now()
    
    # Free up bed
    if hospital:
        hospital.beds_free += 1
    
    # Log activity
    activity = models.Activity(
        time=datetime.now().strftime("%H:%M"),
        patient_id=patient.patient_id,
        hospital=patient.assigned_hospital,
        specialty=patient.specialty_required,
        urgency=patient.triage_level,
        action="DISCHARGED - REFERRAL SENT"
    )
    db.add(activity)
    db.commit()
    
    return {
        "success": True,
        "status": "discharged_with_referral",
        "referral_letter": referral,
        "message": f"Patient #{patient.patient_id} discharged with GP referral"
    }