"""
HSE Multi-Agent Manager - Backend with SQLite Database

Features:
- SQLite database for persistent storage
- OR-Tools constraint programming for patient-hospital assignment
- DISTANCE-BASED OPTIMIZATION (nearest hospital as top priority)
- Configurable LLM endpoint (supports custom fine-tuned models)
- Auto-triage with bilingual support (English/Irish)
- Real-time hospital capacity tracking
- Patient queue management

Run with: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Database imports
from database import engine, get_db
import models

# Import LLMConfig for startup validation
from llm import LLMConfig


# ============================================
# APP SETUP
# ============================================
app = FastAPI(
    title="HSE Multi-Agent Manager",
    description="AI-powered patient triage and hospital optimization with distance-based assignment",
    version="2.0.0"
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
# STARTUP: INITIALIZE DATABASE WITH 5 CORK HOSPITALS
# ============================================
@app.on_event("startup")
def initialize_database():
    """Initialize database and validate LLM config"""
    import os
    
    # Log database path for debugging
    db_path = os.path.abspath("./hse_triage.db")
    print(f"\n🟡 [STARTUP] Database path: {db_path}")
    print(f"🟡 [STARTUP] Database file exists: {os.path.exists(db_path)}")
    print(f"🟡 [STARTUP] Working directory: {os.getcwd()}")
    
    # Validate LLM configuration
    LLMConfig.validate()
    
    db = next(get_db())
    
    # Check if hospitals exist
    if db.query(models.Hospital).count() == 0:
        # 5 Cork-area hospitals with real locations
        hospitals = [
            {
                "id": 0,
                "name": "Cork University Hospital (CUH)",
                "beds_total": 80,
                "beds_free": 15,
                "wait_time": 45,
                "specialties": ["Cardiology", "Neurology", "General", "Trauma", "Oncology", "Maternity"],
                "location": {"lat": 51.8856, "lng": -8.4897},
            },
            {
                "id": 1,
                "name": "Mercy University Hospital",
                "beds_total": 50,
                "beds_free": 12,
                "wait_time": 30,
                "specialties": ["General", "Oncology", "Cardiology"],
                "location": {"lat": 51.8932, "lng": -8.4961},
            },
            {
                "id": 2,
                "name": "South Infirmary Victoria University Hospital",
                "beds_total": 40,
                "beds_free": 18,
                "wait_time": 20,
                "specialties": ["Orthopaedics", "General", "ENT", "Ophthalmology"],
                "location": {"lat": 51.8912, "lng": -8.4823},
            },
            {
                "id": 3,
                "name": "Mallow General Hospital",
                "beds_total": 30,
                "beds_free": 10,
                "wait_time": 15,
                "specialties": ["General", "Orthopaedics", "Geriatrics"],
                "location": {"lat": 52.1345, "lng": -8.6548},
            },
            {
                "id": 4,
                "name": "Bantry General Hospital",
                "beds_total": 25,
                "beds_free": 8,
                "wait_time": 10,
                "specialties": ["General", "Geriatrics", "Palliative Care"],
                "location": {"lat": 51.6838, "lng": -9.4528},
            },
        ]
        
        for h in hospitals:
            db_hospital = models.Hospital(**h)
            db.add(db_hospital)
        
        db.commit()
        hospital_count = db.query(models.Hospital).count()
        print(f"✅ Database initialized with {hospital_count} hospitals")
        print(f"🟢 [STARTUP] Hospitals in database:")
        for h in db.query(models.Hospital).all():
            print(f"    • {h.name}: {h.beds_free}/{h.beds_total} beds available")
    else:
        hospital_count = db.query(models.Hospital).count()
        print(f"✅ Database already initialized with {hospital_count} hospitals")
    
    db.close()


# ============================================
# ROOT ENDPOINT
# ============================================
@app.get("/")
def root():
    """Health check and API info"""
    return {
        "name": "HSE Multi-Agent Manager API",
        "version": "2.0.0",
        "status": "running",
        "llm_provider": "CloudCIX",
        "llm_configured": LLMConfig.is_configured(),
        "llm_model": LLMConfig.MODEL if LLMConfig.is_configured() else "mock",
        "database": "SQLite",
        "optimization": "Distance-based (nearest hospital priority)",
        "hospitals_count": 5,
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
    print(f"📡 LLM: CloudCIX ({LLMConfig.MODEL if LLMConfig.is_configured() else 'mock mode'})")
    print(f"💾 Database: SQLite (hse_database.db)")
    print(f"🏥 Hospitals: 5 Cork-area facilities")
    print(f"📍 Distance-based optimization ENABLED")
    uvicorn.run(app, host="0.0.0.0", port=8000)
