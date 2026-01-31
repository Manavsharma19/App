"""
HSE Multi-Agent Manager - Backend
=================================
Features:
- OR-Tools constraint programming for patient-hospital assignment
- Configurable LLM endpoint (supports custom fine-tuned models)
- Auto-triage with bilingual support (English/Irish)
- Real-time hospital capacity tracking
- Patient queue management

Run with: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ortools.sat.python import cp_model
import httpx
import json
import os
from datetime import datetime
from typing import Optional
from enum import Enum

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
# DATA MODELS
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
# HOSPITAL DATA (In production, we will use a database)
# ============================================

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
    },
]

# Patient queues
pending_patients = []  # Awaiting assignment
assigned_patients = []  # Already assigned
patient_counter = 100

# Activity log
activity_log = []


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
            
            # OpenAI-compatible format (most frameworks use this)
            payload = {
                "model": LLMConfig.CUSTOM_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Patient says: {symptoms}"}
                ],
                "temperature": 0.3,  # Lower = more consistent
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
                    # OpenAI format
                    content = data["choices"][0]["message"]["content"]
                elif "content" in data:
                    # Anthropic format
                    content = data["content"][0]["text"]
                elif "response" in data:
                    # Simple format
                    content = data["response"]
                elif "text" in data:
                    # Another common format
                    content = data["text"]
                else:
                    # Try to find any string response
                    content = str(data)
                
                return json.loads(content)
                
            except httpx.HTTPError as e:
                print(f"Custom LLM error: {e}")
                # Fall back to mock if custom fails
                return mock_triage(symptoms)
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}, content: {content}")
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
               any(word in lower for word in ["tá", "agus", "mé", "mo "])
    
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

def optimize_patient_assignment() -> dict:
    """
    Use OR-Tools CP-SAT solver to optimally assign pending patients to hospitals.
    
    Objective: Minimize total cost (wait_time × urgency + capacity penalty)
    
    Constraints:
    - Each patient assigned to exactly one hospital
    - Hospital capacity not exceeded
    - Specialty requirements must be met
    """
    
    if not pending_patients:
        return {"status": "no_patients", "assignments": []}
    
    model = cp_model.CpModel()
    
    num_patients = len(pending_patients)
    num_hospitals = len(hospitals)
    
    # ===== DECISION VARIABLES =====
    # x[p, h] = 1 if patient p is assigned to hospital h
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
            sum(x[p, h] for p in range(num_patients)) <= hospitals[h]["beds_free"]
        )
    
    # ===== CONSTRAINT 3: Specialty must match =====
    for p in range(num_patients):
        patient = pending_patients[p]
        specialty_needed = patient["specialty_required"].lower()
        
        for h in range(num_hospitals):
            hospital = hospitals[h]
            
            # Check if hospital has the required specialty
            has_specialty = any(
                specialty_needed in s.lower() or s.lower() in specialty_needed
                for s in hospital["specialties"]
            ) or specialty_needed == "general"
            
            if not has_specialty:
                # Hard constraint: cannot assign to this hospital
                model.Add(x[p, h] == 0)
    
    # ===== OBJECTIVE: Minimize total cost =====
    cost_terms = []
    
    for p in range(num_patients):
        patient = pending_patients[p]
        urgency = patient["triage_level"]
        
        # Urgency weight: more urgent patients have higher weight
        urgency_weight = {1: 100, 2: 50, 3: 20, 4: 5, 5: 1}.get(urgency, 10)
        
        for h in range(num_hospitals):
            hospital = hospitals[h]
            
            # Cost components:
            # 1. Wait time (weighted by urgency)
            wait_cost = hospital["wait_time"] * urgency_weight
            
            # 2. Capacity utilization penalty (prefer less crowded hospitals)
            if hospital["beds_total"] > 0:
                occupancy_ratio = (hospital["beds_total"] - hospital["beds_free"]) / hospital["beds_total"]
                capacity_penalty = int(occupancy_ratio * 100)
            else:
                capacity_penalty = 1000
            
            # Total cost for this assignment
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
                    
                    # Create assignment record
                    assignment = {
                        "patient_id": patient["patient_id"],
                        "patient": patient,
                        "hospital_id": h,
                        "hospital_name": hospital["name"],
                        "reason": f"{hospital['beds_free']} beds available, {hospital['wait_time']} min wait, {patient['specialty_required']} department"
                    }
                    assignments.append(assignment)
                    
                    # Update hospital capacity
                    hospitals[h]["beds_free"] -= 1
                    
                    # Add to assigned list
                    assigned_patients.append({
                        **patient,
                        "assigned_hospital": hospital["name"],
                        "assigned_at": datetime.now().isoformat()
                    })
                    
                    # Log activity
                    activity_log.insert(0, {
                        "time": datetime.now().strftime("%H:%M"),
                        "patient_id": patient["patient_id"],
                        "hospital": hospital["name"],
                        "specialty": patient["specialty_required"],
                        "urgency": patient["triage_level"]
                    })
        
        # Clear pending queue
        pending_patients.clear()
        
        return {
            "status": "optimal" if status == cp_model.OPTIMAL else "feasible",
            "objective_value": solver.ObjectiveValue(),
            "assignments": assignments,
            "hospitals": hospitals
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
# API ENDPOINTS
# ============================================

@app.get("/")
def root():
    """Health check and API info"""
    return {
        "name": "HSE Multi-Agent Manager API",
        "version": "1.0.0",
        "status": "running",
        "llm_provider": LLMConfig.PROVIDER,
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


@app.post("/api/triage")
async def triage_patient(patient: PatientInput):
    """
    Triage a patient using the configured LLM.
    Adds patient to pending queue for optimization.
    """
    global patient_counter
    
    # Call LLM for triage
    triage_result = await call_llm(patient.symptoms)
    
    # Create patient record
    patient_counter += 1
    patient_record = {
        "patient_id": patient_counter,
        "patient_name": patient.patient_name,
        "age": patient.age,
        "symptoms": patient.symptoms,
        "detected_language": triage_result.get("detected_language", "English"),
        "translated_symptoms": triage_result.get("translated_symptoms"),
        "triage_level": triage_result.get("triage_level", 4),
        "triage_reason": triage_result.get("triage_reason", "Assessment required"),
        "specialty_required": triage_result.get("specialty_required", "General"),
        "chief_complaint": triage_result.get("chief_complaint", "Not specified"),
        "pain_level": triage_result.get("pain_level"),
        "duration": triage_result.get("duration"),
        "timestamp": datetime.now().isoformat()
    }
    
    # Add to pending queue
    pending_patients.append(patient_record)
    
    return {
        "success": True,
        "patient": patient_record,
        "pending_count": len(pending_patients),
        "message": f"Patient #{patient_counter} triaged as Level {patient_record['triage_level']} - {patient_record['specialty_required']}"
    }


@app.post("/api/optimize")
def run_optimization():
    """
    Run OR-Tools optimization to assign all pending patients to hospitals.
    """
    result = optimize_patient_assignment()
    return result


@app.post("/api/triage-and-assign")
async def triage_and_assign(patient: PatientInput):
    """
    Convenience endpoint: Triage patient AND immediately run optimization.
    Good for real-time single-patient flow.
    """
    # First triage
    triage_response = await triage_patient(patient)
    
    # Then optimize
    optimization_result = optimize_patient_assignment()
    
    # Find this patient's assignment
    patient_assignment = None
    for assignment in optimization_result.get("assignments", []):
        if assignment["patient_id"] == triage_response["patient"]["patient_id"]:
            patient_assignment = assignment
            break
    
    return {
        "triage": triage_response["patient"],
        "assignment": patient_assignment,
        "optimization_status": optimization_result["status"],
        "hospitals": hospitals
    }


@app.get("/api/hospitals")
def get_hospitals():
    """Get current status of all hospitals"""
    return {
        "hospitals": hospitals,
        "total_beds_free": sum(h["beds_free"] for h in hospitals),
        "total_beds": sum(h["beds_total"] for h in hospitals)
    }


@app.patch("/api/hospitals/{hospital_id}")
def update_hospital(hospital_id: int, update: HospitalUpdate):
    """Update hospital capacity or wait time"""
    for h in hospitals:
        if h["id"] == hospital_id:
            if update.beds_free is not None:
                h["beds_free"] = update.beds_free
            if update.wait_time is not None:
                h["wait_time"] = update.wait_time
            return {"success": True, "hospital": h}
    
    raise HTTPException(status_code=404, detail="Hospital not found")


@app.get("/api/patients")
def get_patients():
    """Get all patient queues"""
    return {
        "pending": pending_patients,
        "assigned": assigned_patients[-20:],  # Last 20 assigned
        "pending_count": len(pending_patients),
        "assigned_count": len(assigned_patients)
    }


@app.get("/api/activity")
def get_activity():
    """Get recent activity log"""
    return {
        "activity": activity_log[:20]  # Last 20 activities
    }


@app.post("/api/config/llm")
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


@app.get("/api/config/llm")
def get_llm_config():
    """Get current LLM configuration"""
    return {
        "provider": LLMConfig.PROVIDER,
        "custom_endpoint": LLMConfig.CUSTOM_ENDPOINT,
        "custom_model": LLMConfig.CUSTOM_MODEL_NAME,
        "has_anthropic_key": bool(LLMConfig.ANTHROPIC_API_KEY),
        "has_openai_key": bool(LLMConfig.OPENAI_API_KEY)
    }


@app.post("/api/reset")
def reset_all():
    """Reset all data for demo"""
    global patient_counter, pending_patients, assigned_patients, activity_log
    
    patient_counter = 100
    pending_patients = []
    assigned_patients = []
    activity_log = []
    
    # Reset hospital beds
    hospitals[0]["beds_free"] = 12
    hospitals[1]["beds_free"] = 18
    hospitals[2]["beds_free"] = 8
    
    return {"success": True, "message": "All data reset"}


# ============================================
# RUN SERVER
# ============================================

if __name__ == "__main__":
    import uvicorn
    print("🏥 Starting HSE Multi-Agent Manager Backend...")
    print(f"📡 LLM Provider: {LLMConfig.PROVIDER}")
    print(f"🔗 Custom Endpoint: {LLMConfig.CUSTOM_ENDPOINT}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
