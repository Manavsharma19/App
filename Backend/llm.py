"""
LLM Integration for HSE Multi-Agent Manager
Uses CloudCIX LLM API
"""

import httpx
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ============================================
# SPECIALTY NAME NORMALIZATION (English/Irish)
# ============================================
SPECIALTY_MAPPING = {
    # Map any Irish/variant names to English specialties
    "ginearálta": "General",
    "general": "General",
    "cardieolaíochta": "Cardiology",
    "cardiology": "Cardiology",
    "neurology": "Neurology",
    "néareolaíochta": "Neurology",
    "orthopaedics": "Orthopaedics",
    "ortopéidics": "Orthopaedics",
    "trauma": "Trauma",
    "maternity": "Maternity",
    "leanbhaitheach": "Maternity",
    "oncology": "Oncology",
    "onceolaigí": "Oncology",
    "ent": "ENT",
    "ophthalmology": "Ophthalmology",
    "súileolaiochta": "Ophthalmology",
    "geriatrics": "Geriatrics",
    "palliative care": "Palliative Care",
    "cúram faoina bhás": "Palliative Care",
}

def normalize_specialty(specialty_name: str) -> str:
    """Convert any specialty name (Irish/English/variant) to standard English name"""
    if not specialty_name:
        return "General"
    
    normalized = specialty_name.lower().strip()
    
    # Check exact match
    if normalized in SPECIALTY_MAPPING:
        return SPECIALTY_MAPPING[normalized]
    
    # Check partial match (contains)
    for key, value in SPECIALTY_MAPPING.items():
        if key in normalized or normalized in key:
            return value
    
    # Default to General if unrecognized
    print(f"⚠️ [LLM] Unrecognized specialty '{specialty_name}', defaulting to 'General'")
    return "General"


# ============================================
# LLM CONFIGURATION
# ============================================
class LLMConfig:
    """
    CloudCIX LLM Configuration
    Set these in your .env file
    """
    ENDPOINT = os.getenv("LLM_ENDPOINT", "https://ml-openai.cloudcix.com")
    API_KEY = os.getenv("LLM_API_KEY", "")
    MODEL = os.getenv("LLM_MODEL", "HSEAgent")
    TIMEOUT = 30.0
    
    @classmethod
    def is_configured(cls) -> bool:
        """Check if LLM is properly configured"""
        return bool(cls.ENDPOINT and cls.API_KEY and cls.MODEL)
    
    @classmethod
    def validate(cls):
        """Validate configuration on startup"""
        if not cls.is_configured():
            print("⚠️  WARNING: LLM not configured.")
            print("   Set LLM_ENDPOINT, LLM_API_KEY, and LLM_MODEL in .env")
            return False
        
        print(f"✅ LLM configured:")
        print(f"   Endpoint: {cls.ENDPOINT}")
        print(f"   Model: {cls.MODEL}")
        print(f"   API Key: {'*' * 20}...{cls.API_KEY[-8:] if len(cls.API_KEY) > 8 else '***'}")
        return True


# ============================================
# SYSTEM PROMPTS
# ============================================
TRIAGE_SYSTEM_PROMPT = """You are an HSE (Health Service Executive, Ireland) triage nurse assistant.
Your job is to assess patient symptoms and provide triage information.

Given patient symptoms, you MUST return ONLY valid JSON with no additional text, markdown, or explanation:

{
  "detected_language": "Irish" or "English",
  "translated_symptoms": "English translation if input was Irish, otherwise null",
  "triage_level": 1-5,
  "triage_reason": "Brief clinical explanation for the triage level",
  "specialty_required": "One of: Cardiology, Neurology, General, Orthopaedics, Trauma, Maternity, Oncology, ENT, Ophthalmology, Geriatrics, Palliative Care",
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
# LLM CALL FUNCTION
# ============================================
async def call_llm(symptoms: str, system_prompt: str = None) -> dict:
    """
    Call CloudCIX LLM with custom system prompt.
    
    Args:
        symptoms: Patient input/context
        system_prompt: Custom system prompt (defaults to TRIAGE_SYSTEM_PROMPT)
    
    Raises:
        Exception: If LLM call fails
    """
    if system_prompt is None:
        system_prompt = TRIAGE_SYSTEM_PROMPT
    
    if not LLMConfig.is_configured():
        raise Exception("LLM not configured. Set LLM_ENDPOINT, LLM_API_KEY, and LLM_MODEL in .env")
    
    async with httpx.AsyncClient(timeout=LLMConfig.TIMEOUT) as client:
        # Prepare request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {LLMConfig.API_KEY}"
        }
        
        payload = {
            "model": LLMConfig.MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": symptoms}
            ],
            "temperature": 0.3,
            "max_tokens": 1024
        }
        
        # Build the full API endpoint URL
        api_url = f"{LLMConfig.ENDPOINT.rstrip('/')}/chat/completions"
        
        print(f"🔄 Calling CloudCIX LLM: {api_url}")
        print(f"   Model: {LLMConfig.MODEL}")
        
        # Make request
        response = await client.post(
            api_url,
            headers=headers,
            json=payload
        )
        
        # Check response status
        if response.status_code != 200:
            raise Exception(f"LLM HTTP {response.status_code}: {response.text[:200]}")
        
        # Parse response
        data = response.json()
        print(f"✅ CloudCIX LLM responded: {str(data)[:100]}...")
        
        # Extract content from different response formats
        content = None
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0].get("message", {}).get("content")
        elif "content" in data:
            if isinstance(data["content"], list) and len(data["content"]) > 0:
                content = data["content"][0].get("text")
            else:
                content = data["content"]
        elif "response" in data:
            content = data["response"]
        elif "text" in data:
            content = data["text"]
        
        if not content:
            raise Exception(f"Unable to extract content from LLM response: {data}")
        
        # Parse JSON from content
        result = json.loads(content) if isinstance(content, str) else content
        
        # Normalize specialty name to English
        if "specialty_required" in result:
            result["specialty_required"] = normalize_specialty(result["specialty_required"])
        
        print(f"✅ Triage result: Level {result.get('triage_level')}, {result.get('specialty_required')}")
        return result
