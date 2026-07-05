# main.py  — Soil Chatbot API
# ─────────────────────────────────────────────────────────────────────────────
# Loads: models/soil_imputer.pkl
#        models/soil_model.pkl
#        models/crop_map.pkl
# Run:   uvicorn main:app --reload --port 8000
# ─────────────────────────────────────────────────────────────────────────────
from uuid import uuid4
from datetime import datetime, timedelta
import os
import joblib
import numpy as np
import secrets
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from llama import explain
import threading
import time 
import hashlib
from typing import Optional


load_dotenv()

# ── Supabase connection via REST API (no SDK needed) ─────────────────────────
import httpx

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

def supabase_insert(table: str, data: dict):
    try:
        with httpx.Client() as client:
            response = client.post(
                f"{SUPABASE_URL}/rest/v1/{table}",
                headers=SUPABASE_HEADERS,
                json=data
            )
            return response.json()
    except Exception as e:
        print(f"[Supabase] Insert error on {table}:", str(e))
        return None

def supabase_update(table: str, data: dict, match: dict):
    try:
        params = "&".join([f"{k}=eq.{v}" for k, v in match.items()])
        with httpx.Client() as client:
            response = client.patch(
                f"{SUPABASE_URL}/rest/v1/{table}?{params}",
                headers=SUPABASE_HEADERS,
                json=data
            )
            return response.json()
    except Exception as e:
        print(f"[Supabase] Update error on {table}:", str(e))
        return None

def supabase_select(table: str, match: dict = None):
    try:
        params = ""
        if match:
            params = "?" + "&".join([f"{k}=eq.{v}" for k, v in match.items()])
        with httpx.Client() as client:
            response = client.get(
                f"{SUPABASE_URL}/rest/v1/{table}{params}",
                headers=SUPABASE_HEADERS
            )
            return response.json()
    except Exception as e:
        print(f"[Supabase] Select error on {table}:", str(e))
        return None

print("✅ Supabase REST client ready")

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
    "ph":          {"min": 5.5, "max": 7.0,  "unit": ""},
    "nitrogen":    {"min": 130, "max": 250,   "unit": "mg/kg"},
    "phosphorus":  {"min": 20,  "max": 55,    "unit": "mg/kg"},
    "potassium":   {"min": 90,  "max": 230,   "unit": "mg/kg"},
    "humidity":    {"min": 55,  "max": 90,    "unit": "%"},
    "temperature": {"min": 18,  "max": 35,    "unit": "°C"},
}

# ── In-memory session store ───────────────────────────────────────────────────
# Each session holds the soil report + full conversation history
# Format: { session_id: { "report": {...}, "history": [...] } }
SESSIONS = {}

# ── Gadget (sensor device) tracking ───────────────────────────────────────────
# The physical sensor gadget sends periodic pings so the mobile app
# can show whether it is currently connected.
GADGET_LAST_SEEN = None       # datetime of the most recent ping
GADGET_TIMEOUT   = 15         # seconds — gadget is considered "disconnected" after this silence
GADGET_LATEST    = {          # latest sensor reading submitted via /analyze
    "reading":   None,        # { ph, nitrogen, ... }
    "analysis":  None,        # { fertilizer, crops, soil_health, ... }
    "timestamp": None,        # ISO datetime string
    "farmer_name": None,      # Name of the farmer who submitted this reading
}

@app.post("/gadget/ping")
def gadget_ping():
    """Called periodically by the sensor gadget to announce it is alive."""
    global GADGET_LAST_SEEN
    GADGET_LAST_SEEN = datetime.now()
    farmer_name = GADGET_LATEST.get("farmer_name") or "Unknown"
    print(f"📡 GADGET PING from {farmer_name} at {GADGET_LAST_SEEN.strftime('%H:%M:%S')}")
    return {"status": "ok", "message": "Gadget ping received"}

# ── Gadget status ─────────────────────────────────────────────────────────────
GADGET_STATUS = {
    "status": "offline",
    "battery": 0,
    "last_ping": None
}
# ── Reading trigger flag ──────────────────────────────────────────────────────
READING_REQUESTED = False

@app.post("/trigger-reading")
def trigger_reading():
    global READING_REQUESTED
    READING_REQUESTED = True
    print("[GADGET] Reading requested by mobile app")
    return {"success": True, "message": "Reading will be taken on next gadget check"}

@app.get("/check-trigger")
def check_trigger():
    global READING_REQUESTED
    if READING_REQUESTED:
        READING_REQUESTED = False
        return {"triggered": True}
    return {"triggered": False}

def check_gadget_online():
    while True:
        time.sleep(30)
        if GADGET_STATUS["last_ping"] is None:
            current_status = "offline"
        else:
            last = datetime.fromisoformat(GADGET_STATUS["last_ping"])
            seconds_ago = (datetime.now() - last).total_seconds()
            current_status = "online" if seconds_ago < 60 else "offline"
        
        # Always update Supabase with real status
        supabase_update(
            "gadget_status",
            {"status": current_status},
            {"id": "d56c79a5-20c3-404f-98b0-a6a03f1099d8"}
        )
        print(f"[GADGET] Status check — {current_status}")

# Start background thread
threading.Thread(target=check_gadget_online, daemon=True).start()
print("✅ Gadget monitor started")

@app.post("/gadget-status")
def update_gadget_status(data: dict):
    GADGET_STATUS["status"] = "online"
    GADGET_STATUS["battery"] = data.get("battery", 0)
    GADGET_STATUS["last_ping"] = datetime.now().isoformat()

    # Update ALL rows in gadget_status table (no ID needed)
    supabase_update(
        "gadget_status",
        {
            "status": "online",
            "battery": data.get("battery", 0),
            "last_ping": datetime.now().isoformat()
        },
        {"status": "offline"}  # match any row that exists
    )
    print(f"[GADGET] Heartbeat received — battery: {data.get('battery')}%")
    return {"received": True}

@app.get("/gadget-status")
def get_gadget_status():
    if GADGET_STATUS["last_ping"] is None:
        supabase_update("gadget_status", {"status": "offline", "battery": 0}, {"status": "online"})
        return {"status": "offline", "battery": 0, "last_ping": None}

    last = datetime.fromisoformat(GADGET_STATUS["last_ping"])
    seconds_ago = (datetime.now() - last).total_seconds()
    current_status = "online" if seconds_ago < 60 else "offline"

    # Update Supabase with current real status
    supabase_update(
        "gadget_status",
        {"status": current_status},
        {"status": "online" if current_status == "offline" else "offline"}
    )

    return {
        "status": current_status,
        "battery": GADGET_STATUS["battery"],
        "last_ping": GADGET_STATUS["last_ping"],
        "seconds_ago": int(seconds_ago)
    }


@app.get("/gadget/latest-reading")
def gadget_latest_reading():
    """Returns the most recent sensor reading submitted via /analyze (gadget or manual)."""
    global GADGET_LATEST
    # Ensure we always return a properly formatted response
    response = {
        "reading": GADGET_LATEST.get("reading"),
        "analysis": GADGET_LATEST.get("analysis"),
        "timestamp": GADGET_LATEST.get("timestamp"),
        "farmer_name": GADGET_LATEST.get("farmer_name"),
        "has_data": GADGET_LATEST.get("reading") is not None,
    }
    
    if response["has_data"]:
        print(f"📊 LATEST READING RETRIEVED: {GADGET_LATEST.get('farmer_name')}")
        print(f"   - Fertilizer: {GADGET_LATEST.get('analysis', {}).get('fertilizer')}")
        print(f"   - Session: {GADGET_LATEST.get('analysis', {}).get('session_id')}")
    
    return response
@app.get("/readings/history/{user_id}")
def get_user_history(user_id: str):
    """Returns all sensor readings for a given user, newest first."""
    try:
        readings = supabase_select("sensor_readings", {"user_id": user_id})
        if readings is None:
            return []

        # Sort newest first by created_at
        readings_sorted = sorted(
            readings,
            key=lambda r: r.get("created_at", ""),
            reverse=True
        )

        # Reshape to match what the mobile app expects
        result = []
        for r in readings_sorted:
            result.append({
                "date": r.get("created_at", "")[:10] if r.get("created_at") else None,
                "ph": r.get("ph"),
                "nitrogen": r.get("nitrogen"),
                "phosphorus": r.get("phosphorus"),
                "potassium": r.get("potassium"),
                "humidity": r.get("humidity"),
                "temperature": r.get("temperature"),
                "fertilizer": r.get("fertilizer"),
                "session_id": r.get("session_id"),
            })

        print(f"[History] Returned {len(result)} readings for user {user_id}")
        return result

    except Exception as e:
        print(f"[History] Error fetching history: {str(e)}")
        return []

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
    ph:          float = Field(..., ge=0,    le=14,   example=5.8)
    nitrogen:    float = Field(..., ge=0,    le=1000, example=180)
    phosphorus:  float = Field(..., ge=0,    le=1000, example=25)
    potassium:   float = Field(..., ge=0,    le=1000, example=150)
    humidity:    float = Field(..., ge=0,    le=100,  example=62)
    temperature: float = Field(..., ge=-10,  le=60,   example=28)
    language:    str   = Field("fr", pattern="^(fr|en)$")
    farmer_name: str   = Field("Agriculteur", max_length=60)
    user_id:     Optional[str] = None

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


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

class UserRegister(BaseModel):
    full_name: str = Field(..., min_length=1, max_length=60)
    email: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=6)
    language: str = Field("en", pattern="^(fr|en)$")

class UserLogin(BaseModel):
    email: str
    password: str

@app.post("/auth/register")
def register(user: UserRegister):
    # Check if email already exists
    existing = supabase_select("users", {"email": user.email.lower().strip()})
    if existing and len(existing) > 0:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Save new user
    result = supabase_insert("users", {
        "name": user.full_name.strip(),
        "email": user.email.lower().strip(),
        "password": hash_password(user.password),
        "language": user.language
    })
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to create account")
    
    return {
        "success": True,
        "user_id": result[0]["id"],
        "name": user.full_name.strip(),
        "email": user.email.lower().strip(),
        "language": user.language
    }

@app.post("/auth/login")
def login(user: UserLogin):
    # Find user by email
    existing = supabase_select("users", {"email": user.email.lower().strip()})
    
    if not existing or len(existing) == 0:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    found = existing[0]
    
    # Check password
    if found["password"] != hash_password(user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    return {
        "success": True,
        "user_id": found["id"],
        "name": found["name"],
        "email": found["email"],
        "language": found["language"]
    }
class ReadingUserUpdate(BaseModel):
    session_id: str
    user_id: str

@app.post("/update-reading-user")
def update_reading_user(data: ReadingUserUpdate):
    supabase_update(
        "sensor_readings",
        {"user_id": data.user_id},
        {"session_id": data.session_id}
    )
    supabase_update(
        "chat_sessions",
        {"user_id": data.user_id},
        {"session_id": data.session_id}
    )
    print(f"[Supabase] user_id {data.user_id} linked to session {data.session_id}")
    return {"success": True}

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
        raw = np.array([[
            soil.ph,
            soil.nitrogen,
            soil.phosphorus,
            soil.potassium,
            soil.humidity,
            soil.temperature
        ]])

        # ── Step 2: Preprocess ───────────────────────────────────────────────
        processed = imputer.transform(raw)

        # ── Step 3: Predict fertilizer ───────────────────────────────────────
        fertilizer = str(model.predict(processed)[0])

        # ── Step 4: Get top crop suggestions ─────────────────────────────────
        raw_crops = crop_map.get(fertilizer, [])
        crops = [
            {"rank": i + 1, "name": name}
            for i, name in enumerate(raw_crops)
            if name.strip().lower() != "other"
        ]

        # ── Step 5: Soil health status ───────────────────────────────────────
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

        # ── Step 6: Count issues ─────────────────────────────────────────────
        issues = [
            k for k, v in soil_health.items()
            if v["status"] in ("warning", "critical")
        ]

        # ── Step 7: Generate LLaMA explanation ───────────────────────────────
        explanation = explain(
            fertilizer=fertilizer,
            readings=readings,
            crops=crops,
            soil_health=soil_health,
            language=soil.language,
            farmer_name=soil.farmer_name
        )

       # ── Step 8: Save session in memory ───────────────────────────────────
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
                {
                    "role": "assistant",
                    "content": explanation
                }
            ]
        }

        # ── Step 8b: Update GADGET_LATEST so mobile app can pick it up ───────
        global GADGET_LATEST
        GADGET_LATEST = {
            "reading": readings,
            "analysis": {
                "success":      True,
                "fertilizer":   fertilizer,
                "crops":        crops,
                "soil_health":  soil_health,
                "explanation":  explanation,
                "session_id":   session_id,
                "farmer_name":  soil.farmer_name,
            },
            "timestamp": datetime.now().isoformat(),
            "farmer_name": soil.farmer_name,
        }
        print(f"✅ GADGET READING RECEIVED from {soil.farmer_name}: pH={soil.ph}, N={soil.nitrogen}")
        print(f"   → Prediction: {fertilizer}, Top crop: {crops[0]['name'] if crops else 'N/A'}")

       # ── Step 9: Save sensor reading to Supabase ──────────────────────────
        reading_data = {
    "ph":           soil.ph,
    "nitrogen":     soil.nitrogen,
    "phosphorus":   soil.phosphorus,
    "potassium":    soil.potassium,
    "temperature":  soil.temperature,
    "humidity":     soil.humidity,
    "fertilizer":   fertilizer,
    "crops":        crops,
    "soil_health":  soil_health,
    "explanation":  explanation,
    "session_id":   session_id,
    "user_id":      soil.user_id  
}
        print("[Supabase] Data being sent:", reading_data)
        result = supabase_insert("sensor_readings", reading_data)
        print("[Supabase] Insert result:", result)

        # ── Step 10: Save chat session to Supabase ────────────────────────────
        supabase_insert("chat_sessions", {
    "session_id":   session_id,
    "report":       SESSIONS[session_id]["report"],
    "history":      SESSIONS[session_id]["history"],
    "user_id":      soil.user_id  
})
        print(f"[Supabase] Chat session saved: {session_id}")

        # ── Step 11: Return full structured response to mobile app ────────────
        return {
            "success":      True,
            "session_id":   session_id,
            "farmer_name":  soil.farmer_name,
            "language":     soil.language,
            "fertilizer":   fertilizer,
            "crops":        crops,
            "soil_health":  soil_health,
            "issues_count": len(issues),
            "issue_params": issues,
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

@app.get("/debug/gadget-state")
def debug_gadget_state():
    """Debug endpoint to see the exact state of GADGET_LATEST."""
    global GADGET_LATEST
    return {
        "timestamp": datetime.now().isoformat(),
        "gadget_latest": GADGET_LATEST,
        "gadget_connected": GADGET_LAST_SEEN is not None and (datetime.now() - GADGET_LAST_SEEN).total_seconds() < GADGET_TIMEOUT if GADGET_LAST_SEEN else False,
        "sessions_count": len(SESSIONS),
    }

@app.get("/debug/status")
def debug_status():
    """Debug endpoint to check system status and gadget connection."""
    elapsed = None
    if GADGET_LAST_SEEN:
        elapsed = (datetime.now() - GADGET_LAST_SEEN).total_seconds()
    
    return {
        "system": {
            "models_loaded": MODELS_LOADED,
            "sessions_count": len(SESSIONS),
        },
        "gadget": {
            "connected": GADGET_LAST_SEEN is not None and elapsed < GADGET_TIMEOUT,
            "last_seen": GADGET_LAST_SEEN.isoformat() if GADGET_LAST_SEEN else None,
            "elapsed_seconds": round(elapsed, 1) if elapsed is not None else None,
            "farmer_name": GADGET_LATEST.get("farmer_name"),
            "has_reading": GADGET_LATEST.get("reading") is not None,
            "latest_timestamp": GADGET_LATEST.get("timestamp"),
        },
        "timestamp": datetime.now().isoformat(),
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


# ═══════════════════════════════════════════════════════════════════════════════
# FORGOT PASSWORD  &  READINGS HISTORY
# (Added to support mobile app polish — forgot-password flow + Supabase sync)
# ═══════════════════════════════════════════════════════════════════════════════

# In-memory token store  { token: { email, expires_at } }
# Tokens expire after 1 hour and are deleted on use.
_reset_tokens: dict = {}

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(..., min_length=6)


@app.post("/auth/forgot-password")
def forgot_password(req: ForgotPasswordRequest):
    """
    Generates a password-reset token and stores it in memory for 1 hour.
    Always returns 200 (prevents email enumeration attacks).
    In production: replace the print() with an email-sending call.
    """
    email = req.email.lower().strip()

    existing = supabase_select("users", {"email": email})
    if existing and len(existing) > 0:
        token = secrets.token_urlsafe(32)
        _reset_tokens[token] = {
            "email":      email,
            "expires_at": datetime.now() + timedelta(hours=1),
        }
        # ── TODO: replace these print() lines with your email provider ──────
        print(f"[Password Reset] Token generated for: {email}")
        print(f"[Password Reset] Token (share via email): {token}")
        print(f"[Password Reset] Expires at: {_reset_tokens[token]['expires_at']}")
        # ────────────────────────────────────────────────────────────────────

    # Always respond with success — never reveal whether email exists
    return {
        "success": True,
        "message": "If this email is registered, a reset link has been sent."
    }


@app.post("/auth/reset-password")
def reset_password(req: ResetPasswordRequest):
    """
    Validates the reset token and updates the user's password in Supabase.
    """
    token_data = _reset_tokens.get(req.token)

    if not token_data:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    if datetime.now() > token_data["expires_at"]:
        _reset_tokens.pop(req.token, None)
        raise HTTPException(status_code=400, detail="Reset token has expired. Please request a new one.")

    email = token_data["email"]
    new_hashed = hash_password(req.new_password)

    result = supabase_update("users", {"password": new_hashed}, {"email": email})
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to update password. Please try again.")

    # Invalidate token immediately after use
    _reset_tokens.pop(req.token, None)

    print(f"[Password Reset] Password successfully updated for: {email}")
    return {"success": True, "message": "Password updated successfully. You can now log in."}


@app.get("/readings/history/{user_id}")
def get_readings_history(user_id: str):
    """
    Returns all sensor readings for a given user from Supabase,
    ordered newest-first.  Used by the mobile app to sync history
    across devices instead of relying on local AsyncStorage only.
    """
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(
                f"{SUPABASE_URL}/rest/v1/sensor_readings"
                f"?user_id=eq.{user_id}"
                f"&order=created_at.desc",
                headers=SUPABASE_HEADERS,
            )

        if not response.is_success:
            print(f"[History] Supabase error {response.status_code}: {response.text}")
            raise HTTPException(status_code=502, detail="Could not fetch readings from database")

        rows = response.json()
        if not isinstance(rows, list):
            return []

        # Return only the fields the mobile app needs
        return [
            {
                "date":        r.get("created_at", "")[:10] if r.get("created_at") else "",
                "ph":          r.get("ph",          0),
                "nitrogen":    r.get("nitrogen",    0),
                "phosphorus":  r.get("phosphorus",  0),
                "potassium":   r.get("potassium",   0),
                "humidity":    r.get("humidity",    0),
                "temperature": r.get("temperature", 0),
                "session_id":  r.get("session_id"),
                "fertilizer":  r.get("fertilizer"),
            }
            for r in rows
        ]

    except HTTPException:
        raise
    except Exception as e:
        print(f"[History] Unexpected error for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
