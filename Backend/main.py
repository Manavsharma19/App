"""
HSE Multi-Agent Manager - Backend with SQLite Database

Features:
- SQLite database for persistent storage
- OR-Tools constraint programming for patient-hospital assignment
- Configurable LLM endpoint (supports custom fine-tuned models)
- Auto-triage with bilingual support (English/Irish)
- Real-time hospital capacity tracking
- Patient queue management

Run with: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ortools.sat.python import cp_model
from sqlalchemy.orm import Session
import httpx
import json
import os
from datetime import datetime

# Database imports
from database import engine, get_db
import models

# ============================================
# APP SETUP
# ============================================
app = FastAPI(
    title="HSE Multi-Agent Manager",
    description="AI-powered patient triage and hospital optimization",
    version="1.0.0"
)

# Allow frontend to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables
models.Base.metadata.create_all(bind=engine)


# ============================================
# CONFIGURATION - CHANGE THESE FOR YOUR LLM
# ============================================
class LLMConfig:
    """
    Configure your LLM endpoint here.
    Point this to the fine-tuned model's endpoint.
    """
    # Option 1: Your custom fine-tuned model
    PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "custom", "anthropic", "openai"
    
    # Custom model endpoint (your fine-tuned model)
    CUSTOM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://localhost:5000/v1/chat/completions")
    CUSTOM_API_KEY = os.getenv("LLM_API_KEY", "")  # If your model needs auth
    CUSTOM_MODEL_NAME = os.getenv("LLM_MODEL", "hse-triage-model")  # Your model name
    
    # Anthropic (fallback)
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    
    # OpenAI (fallback)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = "gpt-4o-mini"


# ============================================
# STARTUP: INITIALIZE DATABASE
# ============================================
@app.on_event("startup")
def initialize_database():
    """Initialize database with default hospitals if empty"""
    db = next(get_db())
    
    # Check if hospitals exist
    if db.query(models.Hospital).count() == 0:
        # Default hospitals
        hospitals = [
            {
                "id": 0,
                "name": "Cork University Hospital",
                "beds_total": 50,
                "beds_free": 12,
                "wait_time": 45,
                "specialties": ["Cardiology", "Neurology", "General", "Trauma"],
                "location": {"lat": 51.8856, "lng": -8.4897}
            },
            {
                "id": 1,
                "name": "Mercy University Hospital",
                "beds_total": 40,
                "beds_free": 18,
                "wait_time": 20,
                "specialties": ["General", "Maternity", "Oncology"],
                "location": {"lat": 51.8932, "lng": -8.4961}
            },
            {
                "id": 2,
                "name": "Mallow General Hospital",
                "beds_total": 25,
                "beds_free": 8,
                "wait_time": 15,
                "specialties": ["General", "Orthopaedics"],
                "location": {"lat": 52.1345, "lng": -8.6548}
            }
        ]
        
        for h in hospitals:
            db_hospital = models.Hospital(**h)
            db.add(db_hospital)
        
        db.commit()
        print("✅ Database initialized with default hospitals")
    
    db.close()


# ============================================
# LLM INTEGRATION
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


async def call_llm(symptoms: str) -> dict:
    """
    Call the configured LLM for triage assessment.
    Supports custom endpoints, Anthropic, and OpenAI.
    """
    provider = LLMConfig.PROVIDER
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # ========== CUSTOM MODEL (Your fine-tuned model) ==========
        if provider == "custom":
            headers = {"Content-Type": "application/json"}
            if LLMConfig.CUSTOM_API_KEY:
                headers["Authorization"] = f"Bearer {LLMConfig.CUSTOM_API_KEY}"
            
            payload = {
                "model": LLMConfig.CUSTOM_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Patient says: {symptoms}"}
                ],
                "temperature": 0.3,
                "max_tokens": 1024
            }
            
            try:
                response = await client.post(
                    LLMConfig.CUSTOM_ENDPOINT,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                data = response.json()
                
                # Handle different response formats
                if "choices" in data:
                    content = data["choices"][0]["message"]["content"]
                elif "content" in data:
                    content = data["content"][0]["text"]
                elif "response" in data:
                    content = data["response"]
                elif "text" in data:
                    content = data["text"]
                else:
                    content = str(data)
                
                return json.loads(content)
            
            except (httpx.HTTPError, json.JSONDecodeError) as e:
                print(f"Custom LLM error: {e}")
                return mock_triage(symptoms)
        
        # ========== ANTHROPIC ==========
        elif provider == "anthropic":
            if not LLMConfig.ANTHROPIC_API_KEY:
                return mock_triage(symptoms)
            
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": LLMConfig.ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01"
                },
                json={
                    "model": LLMConfig.ANTHROPIC_MODEL,
                    "max_tokens": 1024,
                    "system": TRIAGE_SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": f"Patient says: {symptoms}"}
                    ]
                }
            )
            response.raise_for_status()
            data = response.json()
            return json.loads(data["content"][0]["text"])
        
        # ========== OPENAI ==========
        elif provider == "openai":
            if not LLMConfig.OPENAI_API_KEY:
                return mock_triage(symptoms)
            
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {LLMConfig.OPENAI_API_KEY}"
                },
                json={
                    "model": LLMConfig.OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Patient says: {symptoms}"}
                    ]
                }
            )
            response.raise_for_status()
            data = response.json()
            return json.loads(data["choices"][0]["message"]["content"])
        
        # ========== MOCK (Fallback) ==========
        else:
            return mock_triage(symptoms)


def mock_triage(symptoms: str) -> dict:
    """Fallback mock triage when no LLM is available."""
    lower = symptoms.lower()
    
    # Detect Irish
    is_irish = any(c in symptoms for c in "áéíóúÁÉÍÓÚ") or \
               any(word in lower for word in ["tá", "agus", "mé", "mo"])
    
    # Detect conditions
    has_chest = any(word in lower for word in ["chest", "chliabhrach", "heart", "croí"])
    has_head = any(word in lower for word in ["head", "ceann", "vision", "dizzy"])
    has_broken = any(word in lower for word in ["broke", "fracture", "wrist", "ankle", "fell"])
    has_breath = "breath" in lower or "anáil" in lower
    
    # Default values
    specialty = "General"
    level = 4
    reason = "Standard assessment required"
    complaint = "General symptoms"
    pain = None
    translation = None
    
    if has_chest:
        specialty = "Cardiology"
        level = 1 if has_breath else 2
        reason = "Chest pain with breathing difficulty - immediate evaluation" if has_breath else "Chest pain requires urgent cardiac evaluation"
        complaint = "Chest pain, weakness"
        pain = 7
        if is_irish:
            translation = "Chest pain and feeling weak"
    elif has_head:
        specialty = "Neurology"
        level = 2 if "vision" in lower else 3
        reason = "Headache with visual disturbance" if "vision" in lower else "Headache requires assessment"
        complaint = "Severe headache"
        pain = 6
    elif has_broken:
        specialty = "Orthopaedics"
        level = 3
        reason = "Suspected fracture requires imaging"
        complaint = "Suspected fracture"
        pain = 8
    
    return {
        "detected_language": "Irish" if is_irish else "English",
        "translated_symptoms": translation,
        "triage_level": level,
        "triage_reason": reason,
        "specialty_required": specialty,
        "chief_complaint": complaint,
        "pain_level": pain,
        "duration": None
    }


# ============================================
# OR-TOOLS OPTIMIZATION
# ============================================
def optimize_patient_assignment(db: Session) -> dict:
    """
    Use OR-Tools CP-SAT solver to optimally assign pending patients to hospitals.
    Objective: Minimize total cost (wait_time × urgency + capacity penalty)
    Constraints:
    - Each patient assigned to exactly one hospital
    - Hospital capacity not exceeded
    - Specialty requirements must be met
    """
    # Get pending patients and hospitals from database
    pending_patients = db.query(models.Patient).filter(models.Patient.status == "pending").all()
    hospitals = db.query(models.Hospital).all()
    
    if not pending_patients:
        return {"status": "no_patients", "assignments": []}
    
    model = cp_model.CpModel()
    num_patients = len(pending_patients)
    num_hospitals = len(hospitals)
    
    # ===== DECISION VARIABLES =====
    x = {}
    for p in range(num_patients):
        for h in range(num_hospitals):
            x[p, h] = model.NewBoolVar(f"patient_{p}_to_hospital_{h}")
    
    # ===== CONSTRAINT 1: Each patient assigned to exactly one hospital =====
    for p in range(num_patients):
        model.Add(sum(x[p, h] for h in range(num_hospitals)) == 1)
    
    # ===== CONSTRAINT 2: Hospital capacity not exceeded =====
    for h in range(num_hospitals):
        model.Add(
            sum(x[p, h] for p in range(num_patients)) <= hospitals[h].beds_free
        )
    
    # ===== CONSTRAINT 3: Specialty must match =====
    for p in range(num_patients):
        patient = pending_patients[p]
        specialty_needed = patient.specialty_required.lower()
        
        for h in range(num_hospitals):
            hospital = hospitals[h]
            has_specialty = any(
                specialty_needed in s.lower() or s.lower() in specialty_needed
                for s in hospital.specialties
            ) or specialty_needed == "general"
            
            if not has_specialty:
                model.Add(x[p, h] == 0)
    
    # ===== OBJECTIVE: Minimize total cost =====
    cost_terms = []
    for p in range(num_patients):
        patient = pending_patients[p]
        urgency = patient.triage_level
        urgency_weight = {1: 100, 2: 50, 3: 20, 4: 5, 5: 1}.get(urgency, 10)
        
        for h in range(num_hospitals):
            hospital = hospitals[h]
            wait_cost = hospital.wait_time * urgency_weight
            
            if hospital.beds_total > 0:
                occupancy_ratio = (hospital.beds_total - hospital.beds_free) / hospital.beds_total
                capacity_penalty = int(occupancy_ratio * 100)
            else:
                capacity_penalty = 1000
            
            total_cost = wait_cost + capacity_penalty
            cost_terms.append(x[p, h] * total_cost)
    
    model.Minimize(sum(cost_terms))
    
    # ===== SOLVE =====
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)
    
    # ===== EXTRACT RESULTS =====
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        assignments = []
        
        for p in range(num_patients):
            for h in range(num_hospitals):
                if solver.Value(x[p, h]) == 1:
                    patient = pending_patients[p]
                    hospital = hospitals[h]
                    
                    # Update patient record
                    patient.status = "assigned"
                    patient.assigned_hospital = hospital.name
                    patient.assigned_at = datetime.now()
                    
                    # Update hospital capacity
                    hospital.beds_free -= 1
                    
                    # Log activity
                    activity = models.Activity(
                        time=datetime.now().strftime("%H:%M"),
                        patient_id=patient.patient_id,
                        hospital=hospital.name,
                        specialty=patient.specialty_required,
                        urgency=patient.triage_level
                    )
                    db.add(activity)
                    
                    # Create assignment record
                    assignment = {
                        "patient_id": patient.patient_id,
                        "patient": patient.to_dict(),
                        "hospital_id": hospital.id,
                        "hospital_name": hospital.name,
                        "reason": f"{hospital.beds_free + 1} beds available, {hospital.wait_time} min wait, {patient.specialty_required} department"
                    }
                    assignments.append(assignment)
        
        db.commit()
        
        return {
            "status": "optimal" if status == cp_model.OPTIMAL else "feasible",
            "objective_value": solver.ObjectiveValue(),
            "assignments": assignments,
            "hospitals": [h.to_dict() for h in hospitals]
        }
    
    elif status == cp_model.INFEASIBLE:
        return {
            "status": "infeasible",
            "message": "No valid assignment possible. Check hospital capacities and specialty requirements.",
            "assignments": []
        }
    else:
        return {
            "status": "no_solution",
            "message": "Solver could not find a solution in time.",
            "assignments": []
        }


# ============================================
# ROOT ENDPOINT
# ============================================
@app.get("/")
def root():
    """Health check and API info"""
    return {
        "name": "HSE Multi-Agent Manager API",
        "version": "1.0.0",
        "status": "running",
        "llm_provider": LLMConfig.PROVIDER,
        "database": "SQLite",
        "endpoints": [
            "POST /api/triage - Triage a patient",
            "POST /api/optimize - Run OR-Tools optimization",
            "GET /api/hospitals - Get hospital status",
            "GET /api/patients - Get patient queues",
            "GET /api/activity - Get activity log",
            "POST /api/config/llm - Update LLM configuration",
            "POST /api/reset - Reset all data"
        ]
    }


# ============================================
# INCLUDE ROUTERS
# ============================================
from routes import router
app.include_router(router)


# ============================================
# RUN SERVER
# ============================================
if __name__ == "__main__":
    import uvicorn
    print("🏥 Starting HSE Multi-Agent Manager Backend...")
    print(f"📡 LLM Provider: {LLMConfig.PROVIDER}")
    print(f"💾 Database: SQLite (hse_triage.db)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
