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

# Import from dedicated modules (no circular imports)
from llm import call_llm, LLMConfig, DISCHARGE_FORM_PROMPT, REFERRAL_LETTER_PROMPT
from optimization import optimize_patient_assignment

# ============================================
# PYDANTIC MODELS (API Request/Response)
# ============================================
class PatientInput(BaseModel):
    symptoms: str
    patient_name: Optional[str] = None
    age: Optional[int] = None
    location: Optional[dict] = None  # {"lat": float, "lng": float}

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
    Patient appears as INCOMING in nurse portal - nurse must confirm admission.
    
    Flow:
    1. Create incoming patient record
    2. Run LLM triage assessment
    3. Patient appears as "incoming" in nurse portal
    4. Nurse clicks "Confirm Admission" button
    5. OR-Tools optimization assigns to best hospital
    """
    print(f"\n🟡 [SERVER] Triage endpoint called")
    print(f"🟡 [SERVER] Received patient input: {patient}")
    print(f"🟡 [SERVER] Symptoms: {patient.symptoms[:50]}...")
    print(f"🟡 [SERVER] Location: {patient.location}")
    
    # Call LLM for triage
    print(f"🟡 [SERVER] Calling LLM for triage...")
    triage_result = await call_llm(patient.symptoms)
    print(f"🟢 [SERVER] LLM returned: {triage_result}")
    
    # Get next patient ID
    max_id = db.query(models.Patient.patient_id).order_by(models.Patient.patient_id.desc()).first()
    patient_id = (max_id[0] + 1) if max_id else 100
    print(f"🟡 [SERVER] Generated patient_id: {patient_id}")
    
    # Create patient record as INCOMING (waiting for nurse confirmation)
    db_patient = models.Patient(
        patient_id=patient_id,
        patient_name=patient.patient_name,
        age=patient.age,
        symptoms=patient.symptoms,
        location=patient.location,
        detected_language=triage_result.get("detected_language", "English"),
        translated_symptoms=triage_result.get("translated_symptoms"),
        triage_level=triage_result.get("triage_level", 4),
        triage_reason=triage_result.get("triage_reason", "Assessment required"),
        specialty_required=triage_result.get("specialty_required", "General"),
        chief_complaint=triage_result.get("chief_complaint", "Not specified"),
        pain_level=triage_result.get("pain_level"),
        duration=triage_result.get("duration"),
        status="incoming"  # Patient is now INCOMING - waiting for nurse to confirm admission
    )
    
    print(f"🟡 [SERVER] Created patient object, adding to DB...")
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    print(f"🟢 [SERVER] Patient #{patient_id} saved to DB with status='incoming'")
    
    response = {
        "success": True,
        "patient": db_patient.to_dict(),
        "message": f"Patient #{patient_id} triaged as Level {db_patient.triage_level}. Please proceed to the reception desk. A nurse will confirm your admittance shortly."
    }
    print(f"🟢 [SERVER] Returning response - patient is now INCOMING and visible to nurses")
    return response


@router.post("/optimize")
def run_optimization(db: Session = Depends(get_db)):
    """
    Run OR-Tools optimization to assign all pending patients to hospitals.
    """
    result = optimize_patient_assignment(db)
    return result


@router.post("/patients/{patient_id}/confirm-admission")
def confirm_admission(patient_id: int, db: Session = Depends(get_db)):
    """
    Nurse confirms admission of incoming patient.
    - For NEW patients from intake: Runs optimization to assign them to the best hospital
    - For TRANSFERRED patients: Just accepts them (no re-optimization)
    """
    print(f"\n🟡 [SERVER] Confirm admission called for patient #{patient_id}")
    
    # Find the incoming patient
    patient = db.query(models.Patient).filter(models.Patient.patient_id == patient_id).first()
    
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient #{patient_id} not found")
    
    if patient.status != "incoming":
        raise HTTPException(status_code=400, detail=f"Patient #{patient_id} is not incoming (current status: {patient.status})")
    
    print(f"🟡 [SERVER] Patient #{patient_id} status: {patient.status}")
    print(f"🟡 [SERVER] Transferred from: {patient.transferred_from}")
    
    # Check if this is a TRANSFERRED patient (already assigned, just needs acceptance at new hospital)
    if patient.transferred_from:
        print(f"🟡 [SERVER] This is a TRANSFERRED patient - skipping optimization, just accepting...")
        patient.status = "assigned"
        patient.assigned_at = datetime.now()
        db.commit()
        db.refresh(patient)
        
        print(f"🟢 [SERVER] Patient #{patient_id} accepted at {patient.assigned_hospital}")
        return {
            "success": True,
            "status": "admitted",
            "patient": patient.to_dict(),
            "hospital": patient.assigned_hospital,
            "message": f"Patient #{patient_id} from {patient.transferred_from} accepted at {patient.assigned_hospital}"
        }
    
    # For NEW patients (no transferred_from), run optimization to assign
    print(f"🟡 [SERVER] This is a NEW patient - changing status to 'pending' and running optimization...")
    patient.status = "pending"
    db.commit()
    print(f"🟢 [SERVER] Patient #{patient_id} status changed to pending")
    
    # Run optimization to assign this new patient
    print(f"🟡 [SERVER] Running optimization...")
    optimization_result = optimize_patient_assignment(db)
    print(f"🟢 [SERVER] Optimization complete: {optimization_result['status']}")
    print(f"🟢 [SERVER] Assignments made: {len(optimization_result.get('assignments', []))} patients")
    
    # Refresh patient from database to get updated status and hospital
    db.refresh(patient)
    
    if patient.status == "assigned" and patient.assigned_hospital:
        print(f"🟢 [SERVER] Patient #{patient_id} confirmed and assigned to {patient.assigned_hospital}")
        return {
            "success": True,
            "status": "admitted",
            "patient": patient.to_dict(),
            "hospital": patient.assigned_hospital,
            "message": f"Patient #{patient_id} admitted. Assigned to {patient.assigned_hospital}"
        }
    else:
        print(f"🔴 [SERVER] Patient #{patient_id} status after optimization: {patient.status}, hospital: {patient.assigned_hospital}")
        return {
            "success": False,
            "status": "not_admitted",
            "message": f"Could not assign patient to a hospital. Status: {patient.status}",
            "optimization_status": optimization_result["status"]
        }


@router.post("/triage-and-assign")
async def triage_and_assign(patient: PatientInput, db: Session = Depends(get_db)):
    """
    ⚠️ DEPRECATED: Use /api/triage followed by /api/optimize instead.
    This endpoint bypasses the pending queue. Use the nurse portal workflow instead.
    """
    raise HTTPException(
        status_code=410, 
        detail="This endpoint is deprecated. Use /api/triage for intake, then /api/optimize from nurse portal."
    )


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
    """Get all patient queues (global view)"""
    print(f"\n🟡 [SERVER] GET /api/patients called")
    
    pending = db.query(models.Patient).filter(models.Patient.status == "pending").all()
    assigned = db.query(models.Patient).filter(models.Patient.status == "assigned").order_by(models.Patient.assigned_at.desc()).limit(20).all()
    incoming = db.query(models.Patient).filter(models.Patient.status == "incoming").order_by(models.Patient.timestamp.desc()).all()
    
    print(f"🟡 [SERVER] Query results:")
    print(f"  - Pending: {len(pending)} patients")
    print(f"  - Assigned: {len(assigned)} patients")
    print(f"  - Incoming: {len(incoming)} patients")
    
    for p in pending:
        print(f"    • Pending patient #{p.patient_id}: assigned_hospital={p.assigned_hospital}, status={p.status}")
    for p in assigned:
        print(f"    • Assigned patient #{p.patient_id}: hospital={p.assigned_hospital}")
    
    response = {
        "pending": [p.to_dict() for p in pending],
        "assigned": [p.to_dict() for p in assigned],
        "incoming": [p.to_dict() for p in incoming],
        "pending_count": len(pending),
        "assigned_count": db.query(models.Patient).filter(models.Patient.status == "assigned").count(),
        "incoming_count": len(incoming)
    }
    
    print(f"🟢 [SERVER] Returning: {response['pending_count']} pending, {response['assigned_count']} assigned, {response['incoming_count']} incoming")
    return response


@router.get("/debug/database")
def debug_database(db: Session = Depends(get_db)):
    """Debug endpoint: Check what's in the database"""
    import os
    
    print(f"\n🟡 [DEBUG] Database contents check")
    
    all_patients = db.query(models.Patient).all()
    all_hospitals = db.query(models.Hospital).all()
    all_activities = db.query(models.Activity).all()
    
    print(f"🟡 [DEBUG] Total patients in DB: {len(all_patients)}")
    print(f"🟡 [DEBUG] Total hospitals in DB: {len(all_hospitals)}")
    print(f"🟡 [DEBUG] Total activities in DB: {len(all_activities)}")
    
    # Group patients by status
    status_counts = {}
    for p in all_patients:
        status = p.status
        if status not in status_counts:
            status_counts[status] = 0
        status_counts[status] += 1
        print(f"  • Patient #{p.patient_id}: status={p.status}, hospital={p.assigned_hospital}, specialty={p.specialty_required}")
    
    return {
        "working_directory": os.getcwd(),
        "database_path": os.path.abspath("./hse_triage.db"),
        "total_patients": len(all_patients),
        "status_breakdown": status_counts,
        "patients": [p.to_dict() for p in all_patients],
        "hospitals": [h.to_dict() for h in all_hospitals],
        "activities_count": len(all_activities)
    }


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    import os
    try:
        # Test database connection
        hospital_count = db.query(models.Hospital).count()
        patient_count = db.query(models.Patient).count()
        
        db_path = os.path.abspath("./hse_triage.db")
        db_exists = os.path.exists(db_path)
        
        return {
            "status": "healthy",
            "database": "connected",
            "hospital_count": hospital_count,
            "patient_count": patient_count,
            "database_path": db_path,
            "database_file_exists": db_exists
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
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


class TransferRequest(BaseModel):
    patient_id: int
    from_hospital: str
    to_hospital: str
    specialty: str
    urgency: str
    reason: str
    transport_method: Optional[str] = None
    medical_status: Optional[str] = None


class DischargeToGPRequest(BaseModel):
    patient_id: int
    hospital_id: int
    reason: str
    follow_up_timeline: Optional[str] = "Within 1-2 weeks"
    recommendations: Optional[list] = None


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
    
    # Use LLM with discharge form prompt
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
    
    # Use LLM with referral letter prompt
    try:
        result = await call_llm(context, system_prompt=REFERRAL_LETTER_PROMPT)
        # Ensure result has letter_body field
        if isinstance(result, dict) and 'letter_body' not in result:
            result['letter_body'] = result.get('letter', str(result))
        return result
    except:
        return mock_referral_letter(patient_data, nurse_verdict)


def mock_referral_letter(patient_data: dict, nurse_verdict: str) -> dict:
    """Fallback referral letter generator."""
    letter_date = datetime.now().strftime("%d %B %Y")
    patient_name = patient_data.get('patient_name', 'Patient Name')
    hospital = patient_data.get('assigned_hospital', 'HSE Hospital')
    age = patient_data.get('age', 'N/A')
    symptoms = patient_data.get('symptoms', 'Not specified')
    
    letter_body = f"""DATE: {letter_date}

RE: {patient_name}, Age {age}

Dear General Practitioner,

This is to formally notify you of the recent hospital admission and subsequent discharge of the above-named patient who was under our care.

REASON FOR PRESENTATION:
{symptoms}

HOSPITAL ADMISSION AND CLINICAL COURSE:
The patient was admitted to {hospital} and received appropriate care and monitoring during their hospital stay. A comprehensive clinical assessment was undertaken with relevant investigations and treatment as required.

CLINICAL ASSESSMENT:
{nurse_verdict}

CURRENT STATUS:
The patient has been assessed as suitable for discharge and is currently in a stable condition. All acute medical issues have been addressed appropriately during the hospital admission.

DISCHARGE RECOMMENDATIONS:
• Monitor vital signs regularly
• Ensure continued compliance with prescribed medications
• Attend outpatient follow-up if required
• Return to hospital if symptoms worsen or new concerns arise

FOLLOW-UP INSTRUCTIONS:
Please arrange a follow-up appointment with the patient within 1-2 weeks. Should you have any queries regarding the patient's care or require further information, please do not hesitate to contact us.

We look forward to continued collaboration in the care of this patient.

Yours sincerely,

HSE Discharge Team
{hospital}"""
    
    return {
        "letter_date": letter_date,
        "gp_salutation": "Dear General Practitioner",
        "patient_info": f"{patient_name}, Age {age}",
        "reason_for_presentation": symptoms,
        "treatment_provided": "Patient received appropriate care and monitoring during hospital stay",
        "current_status": "Patient discharged in stable condition",
        "recommendations": ["Monitor vital signs", "Ensure medication compliance", "Return if symptoms worsen"],
        "follow_up_timeline": "Within 1-2 weeks",
        "letter_body": letter_body,
        "signature": f"HSE {hospital}"
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


@router.post("/discharge/to-gp")
async def discharge_to_gp(request: DischargeToGPRequest, db: Session = Depends(get_db)):
    """
    Discharge patient with GP follow-up only (no hospital transfer).
    Generates a referral letter for the GP.
    """
    patient = db.query(models.Patient).filter(models.Patient.patient_id == request.patient_id).first()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get hospital
    hospital = db.query(models.Hospital).filter(models.Hospital.id == request.hospital_id).first()
    
    # Generate GP referral letter
    patient_data = patient.to_dict()
    referral = await generate_referral_letter(patient_data, request.reason)
    
    # Update patient status
    patient.status = "discharged_gp"
    patient.discharged_at = datetime.now()
    patient.gp_referral = referral
    
    # Free up bed
    if hospital:
        hospital.beds_free += 1
    
    # Log activity
    activity = models.Activity(
        time=datetime.now().strftime("%H:%M"),
        patient_id=patient.patient_id,
        hospital=patient.assigned_hospital or "N/A",
        specialty=patient.specialty_required,
        urgency=patient.triage_level,
        action="DISCHARGED - GP FOLLOW-UP"
    )
    db.add(activity)
    db.commit()
    
    return {
        "success": True,
        "status": "discharged_to_gp",
        "referral_letter": referral,
        "message": f"Patient #{patient.patient_id} discharged with GP follow-up recommendation"
    }


@router.post("/transfer")
async def transfer_patient(request: TransferRequest, db: Session = Depends(get_db)):
    """
    Transfer patient from one hospital to another.
    Creates an 'incoming' patient record at the receiving hospital.
    """
    # Find the patient
    patient = db.query(models.Patient).filter(models.Patient.patient_id == request.patient_id).first()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Get hospitals
    from_hospital = db.query(models.Hospital).filter(models.Hospital.name == request.from_hospital).first()
    to_hospital = db.query(models.Hospital).filter(models.Hospital.name == request.to_hospital).first()
    
    # Free up bed at sending hospital
    if from_hospital:
        from_hospital.beds_free += 1
    
    # Reserve bed at receiving hospital (if available)
    if to_hospital and to_hospital.beds_free > 0:
        to_hospital.beds_free -= 1
    
    # Update patient record for transfer
    patient.transferred_from = request.from_hospital
    patient.assigned_hospital = request.to_hospital
    patient.transfer_reason = request.reason
    patient.specialty_required = request.specialty
    patient.status = "incoming"  # Mark as incoming at new hospital
    patient.assigned_at = datetime.now()
    
    # Log activity at sending hospital
    activity_out = models.Activity(
        time=datetime.now().strftime("%H:%M"),
        patient_id=patient.patient_id,
        hospital=request.from_hospital,
        specialty=patient.specialty_required,
        urgency=patient.triage_level,
        action=f"TRANSFERRED OUT → {request.to_hospital}"
    )
    db.add(activity_out)
    
    # Log activity at receiving hospital
    activity_in = models.Activity(
        time=datetime.now().strftime("%H:%M"),
        patient_id=patient.patient_id,
        hospital=request.to_hospital,
        specialty=request.specialty,
        urgency=patient.triage_level,
        action=f"INCOMING TRANSFER ← {request.from_hospital}"
    )
    db.add(activity_in)
    
    db.commit()
    
    return {
        "success": True,
        "status": "transferred",
        "patient": patient.to_dict(),
        "message": f"Patient #{patient.patient_id} transferred from {request.from_hospital} to {request.to_hospital}",
        "transfer_details": {
            "from": request.from_hospital,
            "to": request.to_hospital,
            "urgency": request.urgency,
            "specialty": request.specialty,
            "reason": request.reason
        }
    }


@router.get("/patients/incoming/{hospital_name}")
def get_incoming_patients(hospital_name: str, db: Session = Depends(get_db)):
    """Get incoming transfer patients for a specific hospital"""
    incoming = db.query(models.Patient).filter(
        models.Patient.status == "incoming",
        models.Patient.assigned_hospital == hospital_name
    ).all()
    
    return {
        "hospital": hospital_name,
        "incoming_patients": [p.to_dict() for p in incoming],
        "count": len(incoming)
    }


@router.post("/patients/{patient_id}/accept")
def accept_incoming_patient(patient_id: int, db: Session = Depends(get_db)):
    """Accept an incoming transfer patient (mark as assigned)"""
    patient = db.query(models.Patient).filter(models.Patient.patient_id == patient_id).first()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if patient.status != "incoming":
        raise HTTPException(status_code=400, detail="Patient is not an incoming transfer")
    
    patient.status = "assigned"
    patient.assigned_at = datetime.now()
    
    # Log activity
    activity = models.Activity(
        time=datetime.now().strftime("%H:%M"),
        patient_id=patient.patient_id,
        hospital=patient.assigned_hospital,
        specialty=patient.specialty_required,
        urgency=patient.triage_level,
        action="TRANSFER ACCEPTED"
    )
    db.add(activity)
    db.commit()
    
    return {
        "success": True,
        "message": f"Patient #{patient_id} accepted",
        "patient": patient.to_dict()
    }