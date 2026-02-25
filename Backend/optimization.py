"""
OR-Tools Optimization for HSE Multi-Agent Manager
Distance-based patient-hospital assignment optimization
"""

import math
from datetime import datetime
from ortools.sat.python import cp_model
from sqlalchemy.orm import Session

import models
from llm import normalize_specialty


# ============================================
# DISTANCE CALCULATION
# ============================================
def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth (in km).
    Used for finding nearest hospital to patient.
    """
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat / 2) ** 2 + \
        math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


# Default patient location (Cork City Centre) when not provided
DEFAULT_PATIENT_LOCATION = {"lat": 51.8985, "lng": -8.4756}


# ============================================
# OR-TOOLS OPTIMIZATION (DISTANCE-BASED)
# ============================================
def optimize_patient_assignment(db: Session) -> dict:
    """
    Use OR-Tools CP-SAT solver to optimally assign pending patients to hospitals.
    
    🎯 DISTANCE IS THE TOP PRIORITY - nearest suitable hospital is preferred.
    
    Objective: Minimize total cost with weighted components:
        Total Cost = (distance × DISTANCE_WEIGHT) + (wait_time × urgency × WAIT_WEIGHT) + (capacity_penalty × CAPACITY_WEIGHT)
    
    Where: DISTANCE_WEIGHT >> WAIT_WEIGHT > CAPACITY_WEIGHT
    
    Constraints:
    - Each patient assigned to exactly one hospital
    - Hospital capacity not exceeded
    - Specialty requirements must be met (HARD constraint)
    """
    print(f"\n🟡 [OPTIMIZATION] Starting optimization...")
    
    # Get pending patients and hospitals from database
    pending_patients = db.query(models.Patient).filter(models.Patient.status == "pending").all()
    hospitals = db.query(models.Hospital).all()
    
    print(f"🟡 [OPTIMIZATION] Pending patients: {len(pending_patients)}")
    print(f"🟡 [OPTIMIZATION] Available hospitals: {len(hospitals)}")
    
    if not pending_patients:
        print(f"🟡 [OPTIMIZATION] No pending patients, returning empty")
        return {"status": "no_patients", "assignments": []}
    
    # DEBUG: Print patient and hospital data
    for p in pending_patients:
        print(f"  Patient #{p.patient_id}: specialty={p.specialty_required}, location={p.location}")
    
    print(f"🟡 [OPTIMIZATION] Hospital details:")
    for h in hospitals:
        print(f"  {h.name}: beds_free={h.beds_free}, specialties={h.specialties}, location={h.location}")
    
    model = cp_model.CpModel()
    num_patients = len(pending_patients)
    num_hospitals = len(hospitals)
    
    # ===== WEIGHT CONFIGURATION =====
    # DISTANCE IS TOP PRIORITY - highest weight
    DISTANCE_WEIGHT = 1000  # Multiplier for distance cost (TOP PRIORITY)
    WAIT_WEIGHT = 10        # Multiplier for wait time cost
    CAPACITY_WEIGHT = 1     # Multiplier for capacity penalty (lowest priority)
    
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
    
    # ===== CONSTRAINT 3: Specialty must match (HARD CONSTRAINT) =====
    specialty_mismatches = 0
    for p in range(num_patients):
        patient = pending_patients[p]
        # Normalize patient specialty to standard English name
        specialty_needed = normalize_specialty(patient.specialty_required).lower()
        
        eligible_hospitals = 0
        for h in range(num_hospitals):
            hospital = hospitals[h]
            has_specialty = any(
                specialty_needed in s.lower() or s.lower() in specialty_needed
                for s in hospital.specialties
            ) or specialty_needed == "general"
            
            if has_specialty:
                eligible_hospitals += 1
            else:
                model.Add(x[p, h] == 0)
                specialty_mismatches += 1
        
        print(f"  Patient #{patient.patient_id} ({specialty_needed}): {eligible_hospitals}/{num_hospitals} hospitals eligible")
        if eligible_hospitals == 0:
            print(f"    🔴 ERROR: NO ELIGIBLE HOSPITALS for specialty '{specialty_needed}'!")
    
    print(f"🟡 [OPTIMIZATION] Specialty constraints applied: {specialty_mismatches} mismatches blocked\n")
    
    # ===== PRE-CALCULATE DISTANCES =====
    distances = {}
    for p in range(num_patients):
        patient = pending_patients[p]
        # Get patient location (use stored location or default)
        patient_loc = patient.location if patient.location else DEFAULT_PATIENT_LOCATION
        
        for h in range(num_hospitals):
            hospital = hospitals[h]
            hospital_loc = hospital.location
            
            # Calculate distance in km using Haversine formula
            dist_km = haversine_distance(
                patient_loc["lat"], patient_loc["lng"],
                hospital_loc["lat"], hospital_loc["lng"]
            )
            distances[p, h] = dist_km
    
    # ===== OBJECTIVE: Minimize total cost (DISTANCE IS TOP PRIORITY) =====
    cost_terms = []
    for p in range(num_patients):
        patient = pending_patients[p]
        urgency = patient.triage_level
        urgency_weight = {1: 100, 2: 50, 3: 20, 4: 5, 5: 1}.get(urgency, 10)
        
        for h in range(num_hospitals):
            hospital = hospitals[h]
            
            # ===== COST COMPONENT 1: DISTANCE (TOP PRIORITY) =====
            distance_cost = int(distances[p, h] * 100) * DISTANCE_WEIGHT
            
            # ===== COST COMPONENT 2: Wait time (weighted by urgency) =====
            wait_cost = hospital.wait_time * urgency_weight * WAIT_WEIGHT
            
            # ===== COST COMPONENT 3: Capacity utilization penalty =====
            if hospital.beds_total > 0:
                occupancy_ratio = (hospital.beds_total - hospital.beds_free) / hospital.beds_total
                capacity_penalty = int(occupancy_ratio * 100) * CAPACITY_WEIGHT
            else:
                capacity_penalty = 1000 * CAPACITY_WEIGHT
            
            total_cost = distance_cost + wait_cost + capacity_penalty
            cost_terms.append(x[p, h] * total_cost)
    
    model.Minimize(sum(cost_terms))
    
    # ===== SOLVE =====
    print(f"🟡 [OPTIMIZATION] Solving CP-SAT model...")
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)
    
    print(f"🟡 [OPTIMIZATION] Solver status: {status}")
    print(f"  OPTIMAL={cp_model.OPTIMAL}, FEASIBLE={cp_model.FEASIBLE}, INFEASIBLE={cp_model.INFEASIBLE}")
    
    # ===== EXTRACT RESULTS =====
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        assignments = []
        
        for p in range(num_patients):
            for h in range(num_hospitals):
                if solver.Value(x[p, h]) == 1:
                    patient = pending_patients[p]
                    hospital = hospitals[h]
                    
                    # Get distance for this assignment
                    dist_km = distances[p, h]
                    
                    # Update patient record
                    patient.status = "assigned"
                    patient.assigned_hospital = hospital.name
                    patient.assigned_at = datetime.now()
                    
                    # Update hospital capacity
                    hospital.beds_free -= 1
                    
                    # Log activity with distance
                    activity = models.Activity(
                        time=datetime.now().strftime("%H:%M"),
                        patient_id=patient.patient_id,
                        hospital=hospital.name,
                        specialty=patient.specialty_required,
                        urgency=patient.triage_level
                    )
                    db.add(activity)
                    
                    # Create assignment record with distance info
                    assignment = {
                        "patient_id": patient.patient_id,
                        "patient": patient.to_dict(),
                        "hospital_id": hospital.id,
                        "hospital_name": hospital.name,
                        "distance_km": round(dist_km, 2),
                        "reason": f"Nearest suitable hospital ({dist_km:.1f} km), {hospital.beds_free + 1} beds, ~{hospital.wait_time} min wait, {patient.specialty_required} dept"
                    }
                    assignments.append(assignment)
                    print(f"🟢 [OPTIMIZATION] Patient #{patient.patient_id} → {hospital.name}")
        
        db.commit()
        
        return {
            "status": "optimal" if status == cp_model.OPTIMAL else "feasible",
            "objective_value": solver.ObjectiveValue(),
            "assignments": assignments,
            "hospitals": [h.to_dict() for h in hospitals]
        }
    
    elif status == cp_model.INFEASIBLE:
        print(f"🔴 [OPTIMIZATION] INFEASIBLE - No valid assignment found!")
        print(f"  Check:")
        print(f"  1. Hospital capacities (total free beds: {sum(h.beds_free for h in hospitals)})")
        print(f"  2. Specialty match (patient needs vs hospital specialties)")
        
        return {
            "status": "infeasible",
            "message": "No valid assignment possible. Check hospital capacities and specialty requirements.",
            "assignments": []
        }
    else:
        print(f"🔴 [OPTIMIZATION] NO SOLUTION - Solver timeout or error")
        return {
            "status": "no_solution",
            "message": "Solver could not find a solution in time.",
            "assignments": []
        }


