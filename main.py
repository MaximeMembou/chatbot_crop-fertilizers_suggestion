# main.py  — Soil Chatbot API
# ─────────────────────────────────────────────────────────────────────────────
# Loads: models/soil_imputer.pkl
#        models/soil_model.pkl
#        models/crop_map.pkl
# Run:   uvicorn main:app --reload --port 8000
# ─────────────────────────────────────────────────────────────────────────────
from uuid import uuid4
from datetime import datetime
import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from llama import explain

load_dotenv()

app = FastAPI(
    title="Soil Chatbot API — Cameroun",
    description="Recommande engrais et cultures à partir des lectures de capteur de sol.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Load your three pkl files once at startup ─────────────────────────────────
MODELS_LOADED = False
MODEL_ERROR   = ""

try:
    imputer   = joblib.load("models/soil_imputer.pkl")
    model     = joblib.load("models/soil_model.pkl")
    crop_map  = joblib.load("models/crop_map.pkl")   # fertilizer → top 3 crops
    MODELS_LOADED = True
    print("✅ soil_imputer.pkl  loaded")
    print("✅ soil_model.pkl    loaded")
    print("✅ crop_map.pkl      loaded")
    print(f"   Fertilizer classes: {list(model.classes_)}")
except Exception as e:
    MODEL_ERROR = str(e)
    print(f"❌ Failed to load models: {MODEL_ERROR}")

# ── Ideal ranges for each sensor parameter (maize baseline) ──────────────────
IDEAL = {
    "ph":          {"min": 6.0, "max": 7.0,  "unit": ""},
    "nitrogen":    {"min": 140, "max": 280,   "unit": "mg/kg"},
    "phosphorus":  {"min": 10,  "max": 40,    "unit": "mg/kg"},
    "potassium":   {"min": 120, "max": 280,   "unit": "mg/kg"},
    "humidity":    {"min": 50,  "max": 70,    "unit": "%"},
    "temperature": {"min": 20,  "max": 35,    "unit": "°C"},
}

# ── In-memory session store ───────────────────────────────────────────────────
# Each session holds the soil report + full conversation history
# Format: { session_id: { "report": {...}, "history": [...] } }
SESSIONS = {}

def get_status(value: float, min_val: float, max_val: float) -> str:
    if min_val <= value <= max_val:
        return "good"
    gap = (
        (min_val - value) / min_val
        if value < min_val
        else (value - max_val) / max_val
    )
    return "critical" if gap > 0.30 else "warning"

# ── Request schema ────────────────────────────────────────────────────────────
class SoilReading(BaseModel):
    ph:          float = Field(..., ge=0,  le=14,  example=5.8)
    nitrogen:    float = Field(..., ge=0,  le=500, example=80)
    phosphorus:  float = Field(..., ge=0,  le=100, example=12)
    potassium:   float = Field(..., ge=0,  le=500, example=90)
    humidity:    float = Field(..., ge=0,  le=100, example=62)
    temperature: float = Field(..., ge=0,  le=60,  example=28)
    language:    str   = Field("fr", pattern="^(fr|en)$")
    farmer_name: str   = Field("Agriculteur", max_length=60)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Soil Chatbot API is running.",
        "usage":   "POST /analyze with soil readings JSON",
        "docs":    "Visit /docs for interactive testing"
    }

@app.get("/health")
def health():
    return {
        "status":          "ok" if MODELS_LOADED else "degraded",
        "models_loaded":   MODELS_LOADED,
        "model_error":     MODEL_ERROR if not MODELS_LOADED else None,
        "fertilizer_classes": list(model.classes_) if MODELS_LOADED else [],
        "version":         "2.0.0"
    }

@app.post("/analyze")
def analyze(soil: SoilReading):

    if not MODELS_LOADED:
        raise HTTPException(
            status_code=503,
            detail=f"Models not ready: {MODEL_ERROR}"
        )
    session_id = str(uuid4())
    try:
        # ── Step 1: Build feature array — order MUST match training ──────────
        # Order: ph, nitrogen, phosphorus, potassium, humidity, temperature
        raw = np.array([[
            soil.ph,
            soil.nitrogen,
            soil.phosphorus,
            soil.potassium,
            soil.humidity,
            soil.temperature
        ]])

        # ── Step 2: Preprocess with soil_imputer.pkl ─────────────────────────
        processed = imputer.transform(raw)

        # ── Step 3: Predict fertilizer with soil_model.pkl ───────────────────
        fertilizer = str(model.predict(processed)[0])

        # ── Step 4: Get top crop suggestions from crop_map.pkl ───────────────
        raw_crops = crop_map.get(fertilizer, [])
        crops = [
            {"rank": i + 1, "name": name}
            for i, name in enumerate(raw_crops)
            if name.strip().lower() != "other"
        ]

        # ── Step 5: Soil health status per parameter ─────────────────────────
        readings = {
            "ph":          soil.ph,
            "nitrogen":    soil.nitrogen,
            "phosphorus":  soil.phosphorus,
            "potassium":   soil.potassium,
            "humidity":    soil.humidity,
            "temperature": soil.temperature,
        }

        soil_health = {}
        for key, val in readings.items():
            r = IDEAL[key]
            soil_health[key] = {
                "value":  val,
                "min":    r["min"],
                "max":    r["max"],
                "unit":   r["unit"],
                "status": get_status(val, r["min"], r["max"])
            }

        # ── Step 6: Count how many parameters are problematic ────────────────
        issues = [
            k for k, v in soil_health.items()
            if v["status"] in ("warning", "critical")
        ]

        # ── Step 7: Generate LLaMA farmer-friendly explanation ───────────────
        explanation = explain(
            fertilizer=fertilizer,
            readings=readings,
            crops=crops,
            soil_health=soil_health,
            language=soil.language,
            farmer_name=soil.farmer_name
        )
       
        SESSIONS[session_id] = {
        "report": {
            "farmer_name":  soil.farmer_name,
            "fertilizer":   fertilizer,
            "crops":        crops,
            "soil_health":  soil_health,
            "explanation":  explanation,
            "readings":     readings,
            "language":     soil.language,
            "created_at":   datetime.now().isoformat()
        },
        "history": [
            # Seed history with the initial report so LLaMA has full context
            {
                "role": "assistant",
                "content": explanation
            }
        ]
    }
        # ── Step 8: Return full structured response to mobile app ────────────
        return {
            "success":      True,
            "session_id":   session_id,        # ← farmer's app stores this
            "farmer_name":  soil.farmer_name,
            "language":     soil.language,

            # PRIMARY: fertilizer always first
            "fertilizer":   fertilizer,

            # SECONDARY: crop suggestions
            "crops":        crops,

            # DETAIL: per-parameter health for chart rendering
            "soil_health":  soil_health,

            # SUMMARY flags for mobile app UI logic
            "issues_count": len(issues),
            "issue_params": issues,

            # NATURAL LANGUAGE: LLaMA-generated explanation
            "explanation":  explanation,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ── New request schema for chat messages ──────────────────────────────────────
class ChatMessage(BaseModel):
    session_id:  str = Field(..., description="ID returned by /analyze")
    message:     str = Field(..., min_length=1, max_length=1000)
    language:    str = Field("fr", pattern="^(fr|en)$")

@app.post("/chat")
def chat(msg: ChatMessage):
    """
    Farmer sends a follow-up question after receiving their soil report.
    The chatbot answers using the full conversation history as context.
    """

    # ── Check session exists ──────────────────────────────────────────────────
    if msg.session_id not in SESSIONS:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please run /analyze first to get a session_id."
        )

    session = SESSIONS[msg.session_id]
    report  = session["report"]

    # ── Add farmer's question to history ──────────────────────────────────────
    session["history"].append({
        "role":    "user",
        "content": msg.message
    })

    # ── Generate context-aware answer from LLaMA ─────────────────────────────
    from llama import chat_reply
    answer = chat_reply(
        history=session["history"],
        report=report,
        language=msg.language
    )

    # ── Add chatbot answer to history ─────────────────────────────────────────
    session["history"].append({
        "role":    "assistant",
        "content": answer
    })

    return {
        "success":    True,
        "session_id": msg.session_id,
        "question":   msg.message,
        "answer":     answer,
        "turn":       len([h for h in session["history"] if h["role"] == "user"])
    }

@app.get("/history/{session_id}")
def get_history(session_id: str):
    """Returns the full conversation history for a session."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = SESSIONS[session_id]
    return {
        "session_id": session_id,
        "farmer":     session["report"].get("farmer_name", "Agriculteur"),
        "turns":      len(session["history"]),
        "history":    session["history"]
    }

# ── New request schema for chat messages ──────────────────────────────────────
class ChatMessage(BaseModel):
    session_id:  str = Field(..., description="ID returned by /analyze")
    message:     str = Field(..., min_length=1, max_length=1000)
    language:    str = Field("fr", pattern="^(fr|en)$")

@app.post("/chat")
def chat(msg: ChatMessage):
    """
    Farmer sends a follow-up question after receiving their soil report.
    The chatbot answers using the full conversation history as context.
    """

    # ── Check session exists ──────────────────────────────────────────────────
    if msg.session_id not in SESSIONS:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please run /analyze first to get a session_id."
        )

    session = SESSIONS[msg.session_id]
    report  = session["report"]

    # ── Add farmer's question to history ──────────────────────────────────────
    session["history"].append({
        "role":    "user",
        "content": msg.message
    })

    # ── Generate context-aware answer from LLaMA ─────────────────────────────
    from llama import chat_reply
    answer = chat_reply(
        history=session["history"],
        report=report,
        language=msg.language
    )

    # ── Add chatbot answer to history ─────────────────────────────────────────
    session["history"].append({
        "role":    "assistant",
        "content": answer
    })

    return {
        "success":    True,
        "session_id": msg.session_id,
        "question":   msg.message,
        "answer":     answer,
        "turn":       len([h for h in session["history"] if h["role"] == "user"])
    }

@app.get("/history/{session_id}")
def get_history(session_id: str):
    """Returns the full conversation history for a session."""
    if session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")
    session = SESSIONS[session_id]
    return {
        "session_id": session_id,
        "farmer":     session["report"].get("farmer_name", "Agriculteur"),
        "turns":      len(session["history"]),
        "history":    session["history"]
    }