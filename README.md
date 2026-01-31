# HSE Multi-Agent Manager

AI-powered patient triage and hospital optimization system for the HSE (Health Service Executive, Ireland).

## Features

- **Auto-Triage**: LLM-powered symptom assessment in English and Irish
- **OR-Tools Optimization**: Constraint programming for optimal patient-hospital assignment
- **Real-time Dashboard**: Live hospital capacity and assignment tracking
- **Bilingual Support**: Automatic language detection and translation
- **Custom LLM Support**: Connect your own fine-tuned model

## Project Structure

```
hse-backend/
├── main.py              # FastAPI backend with OR-Tools
├── intake.html          # Patient Intake UI (chat interface)
├── dashboard.html       # Admin Dashboard UI (monitoring)
├── index.html           # Combined UI (both in one page)
├── requirements.txt     # Python dependencies
└── README.md
```

## Architecture

```
┌─────────────────┐         
│  intake.html    │ (Patient/Nurse facing)
│  Chat Interface │─────┐
└─────────────────┘     │
                        │
┌─────────────────┐     │    ┌─────────────────┐         ┌─────────────────┐
│ dashboard.html  │     ├───►│  FastAPI Backend │ ──────► │   Your LLM      │
│ Admin Dashboard │─────┘    │   (main.py)      │         │ (fine-tuned)    │
└─────────────────┘          └────────┬────────┘         └─────────────────┘
                                      │
                                      ▼
                             ┌─────────────────┐
                             │    OR-Tools     │
                             │   Optimizer     │
                             └─────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd hse-backend
pip install -r requirements.txt
```

### 2. Configure Your LLM

Edit `main.py` or set environment variables:

```bash
# For your custom fine-tuned model:
export LLM_PROVIDER=custom
export LLM_ENDPOINT=http://your-model-server:5000/v1/chat/completions
export LLM_API_KEY=your-api-key
export LLM_MODEL=hse-triage-model

# Or use Anthropic/OpenAI as fallback:
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your-key
```

### 3. Run the Backend

```bash
python main.py
# or
uvicorn main:app --reload --port 8000
```

### 4. Open the Frontend

Serve the files and open in browser:

```bash
python -m http.server 3000
```

Then visit:
- **Patient Intake**: http://localhost:3000/intake.html
- **Admin Dashboard**: http://localhost:3000/dashboard.html
- **Combined View**: http://localhost:3000/index.html

## UI Screens

### Patient Intake (`intake.html`)
- Used by: Patients, Nurses at A&E desk
- Features: Chat interface, symptom input, triage results, hospital assignment
- Supports English and Irish input

### Admin Dashboard (`dashboard.html`)
- Used by: Hospital administrators, HSE coordinators
- Features: Hospital capacity overview, real-time bed tracking, assignment log, pending patients
- Auto-refreshes every 5 seconds

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and API info |
| `/api/triage` | POST | Triage a patient (calls LLM) |
| `/api/optimize` | POST | Run OR-Tools optimization |
| `/api/triage-and-assign` | POST | Triage + optimize in one call |
| `/api/hospitals` | GET | Get hospital status |
| `/api/patients` | GET | Get patient queues |
| `/api/activity` | GET | Get activity log |
| `/api/config/llm` | GET/POST | View/update LLM config |
| `/api/reset` | POST | Reset all data |

## Connecting Your Custom LLM

Your fine-tuned model should accept OpenAI-compatible format:

**Request:**
```json
POST /v1/chat/completions
{
    "model": "hse-triage-model",
    "messages": [
        {"role": "system", "content": "You are an HSE triage nurse..."},
        {"role": "user", "content": "Patient says: Tá pian i mo chliabhrach"}
    ]
}
```

**Expected Response:**
```json
{
    "choices": [
        {
            "message": {
                "content": "{\"detected_language\": \"Irish\", \"triage_level\": 2, ...}"
            }
        }
    ]
}
```

The backend also supports Anthropic and simple `{"response": "..."}` formats.

## OR-Tools Optimization

The optimizer solves:

```
Minimize:
    Σ (wait_time[h] × urgency_weight[p] × x[p,h])

Subject to:
    Σ x[p,h] = 1                    (each patient assigned once)
    Σ x[p,h] ≤ capacity[h]          (hospital capacity)
    x[p,h] = 0 if specialty mismatch (must have right specialty)
```

## Demo Script

1. Click "Test Connection" to connect to backend
2. Try Irish input: "Tá pian i mo chliabhrach agus tá mé ag mothú lag"
3. Watch the triage result (LLM) and assignment (OR-Tools)
4. Try English: "I have a severe headache and blurred vision"
5. Notice different specialty routing

## Hackathon Team

- **Person A**: LLM fine-tuning and prompts
- **Person B**: Frontend (index.html)
- **Person C**: Backend + OR-Tools (main.py)
- **Person D**: Integration and demo
- **Person E**: Presentation

Good luck! 