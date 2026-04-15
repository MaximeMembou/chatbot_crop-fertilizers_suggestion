# llama.py  — LLaMA 3 explanation generator
# ─────────────────────────────────────────────────────────────────────────────
# Connects to LLaMA 3 via Groq (free).
# Called by main.py after model prediction.
# ─────────────────────────────────────────────────────────────────────────────

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── System prompts per language ───────────────────────────────────────────────

SYSTEM_FR = """ ... your French system prompt ... """
SYSTEM_EN = """ ... your English system prompt ... """

def chat_reply(report, language="fr"):
    system = f"""
{'Tu es un conseiller agricole expert pour les agriculteurs du Cameroun.' if language == 'fr' else 'You are an expert agricultural advisor for farmers in Cameroon.'}

{'RÈGLE ABSOLUE : Tu dois répondre UNIQUEMENT en français. Pas un seul mot en anglais.' if language == 'fr' else 'ABSOLUTE RULE: You must respond ONLY in English. Not a single word in French.'}

{'Contexte du sol de cet agriculteur :' if language == 'fr' else "This farmer's soil context:"}
- {'Engrais recommandé' if language == 'fr' else 'Recommended fertilizer'}: {report.get('fertilizer', 'N/A')}
- {'Cultures suggérées' if language == 'fr' else 'Suggested crops'}: {', '.join(c['name'] for c in report.get('crops', []))}
- pH: {report.get('readings', {}).get('ph', 'N/A')}
- {'Azote' if language == 'fr' else 'Nitrogen'}: {report.get('readings', {}).get('nitrogen', 'N/A')} mg/kg
- {'Phosphore' if language == 'fr' else 'Phosphorus'}: {report.get('readings', {}).get('phosphorus', 'N/A')} mg/kg
- {'Potassium' if language == 'fr' else 'Potassium'}: {report.get('readings', {}).get('potassium', 'N/A')} mg/kg
- {'Humidité' if language == 'fr' else 'Humidity'}: {report.get('readings', {}).get('humidity', 'N/A')}%
- {'Température' if language == 'fr' else 'Temperature'}: {report.get('readings', {}).get('temperature', 'N/A')}°C

{'Réponds en maximum 150 mots. Sois concis, pratique et encourageant.' if language == 'fr' else 'Reply in maximum 150 words. Be concise, practical and encouraging.'}
{'Ne mentionne jamais Other comme culture.' if language == 'fr' else 'Never mention Other as a crop.'}
"""


# ── Fertilizer application guides (from your real dataset labels) ─────────────
FERTILIZER_GUIDE = {
    "Urea + DAP": {
        "fr": "Appliquez 50 kg d'Urée + 30 kg de DAP par hectare, mélangés dans le sol avant la plantation.",
        "en": "Apply 50 kg Urea + 30 kg DAP per hectare, mixed into the soil before planting."
    },
    "Urea + DAP + Lime": {
        "fr": "Appliquez 200 kg de chaux d'abord, attendez 2 semaines, puis 50 kg Urée + 30 kg DAP par hectare.",
        "en": "Apply 200 kg lime first, wait 2 weeks, then 50 kg Urea + 30 kg DAP per hectare."
    },
    "NPK 15-15-15": {
        "fr": "Appliquez 100 kg de NPK 15-15-15 par hectare au moment de la plantation.",
        "en": "Apply 100 kg of NPK 15-15-15 per hectare at planting time."
    },
    "NPK 20-10-10": {
        "fr": "Appliquez 100 kg de NPK 20-10-10 par hectare. Idéal pour sols pauvres en azote.",
        "en": "Apply 100 kg of NPK 20-10-10 per hectare. Best for nitrogen-poor soils."
    },
    "DAP + MOP": {
        "fr": "Appliquez 40 kg de DAP + 30 kg de MOP par hectare avant la plantation.",
        "en": "Apply 40 kg DAP + 30 kg MOP per hectare before planting."
    },
    "Balanced NPK": {
        "fr": "Appliquez 80 kg de NPK équilibré par hectare. Votre sol est en bonne santé générale.",
        "en": "Apply 80 kg balanced NPK per hectare. Your soil is in good general health."
    },
    "Urea + MOP": {
        "fr": "Appliquez 50 kg d'Urée + 30 kg de MOP par hectare. Bon pour sols riches en phosphore.",
        "en": "Apply 50 kg Urea + 30 kg MOP per hectare. Good for phosphorus-rich soils."
    },
    "Urea + DAP + MOP": {
        "fr": "Appliquez 40 kg Urée + 25 kg DAP + 25 kg MOP par hectare pour une correction complète.",
        "en": "Apply 40 kg Urea + 25 kg DAP + 25 kg MOP per hectare for complete correction."
    },
}

def _build_issues_text(soil_health: dict, language: str) -> str:
    """Describe which parameters are out of range, in plain language."""
    problems = []
    for param, info in soil_health.items():
        if info["status"] == "good":
            continue
        val  = info["value"]
        mn   = info["min"]
        mx   = info["max"]
        unit = info["unit"]
        label = {
            "ph":          "pH",
            "nitrogen":    "Azote (N)"    if language == "fr" else "Nitrogen (N)",
            "phosphorus":  "Phosphore (P)"if language == "fr" else "Phosphorus (P)",
            "potassium":   "Potassium (K)",
            "humidity":    "Humidité"     if language == "fr" else "Humidity",
            "temperature": "Température"  if language == "fr" else "Temperature",
        }.get(param, param)

        if val < mn:
            word = "faible" if language == "fr" else "low"
        else:
            word = "élevé"  if language == "fr" else "high"

        problems.append(
            f"{label}: {val}{unit} ({word}, idéal {mn}–{mx}{unit})"
            if language == "fr"
            else f"{label}: {val}{unit} ({word}, ideal {mn}–{mx}{unit})"
        )

    if not problems:
        return ("Toutes les valeurs sont dans la plage normale." if language == "fr"
                else "All values are within the normal range.")
    header = "Problèmes détectés :" if language == "fr" else "Issues detected:"
    return header + "\n" + "\n".join(f"  - {p}" for p in problems)


def explain(
    fertilizer: str,
    readings: dict,
    crops: list,
    soil_health: dict,
    language: str = "fr",
    farmer_name: str = "Agriculteur"
) -> str:

    system = SYSTEM_FR if language == "fr" else SYSTEM_EN

    guide = FERTILIZER_GUIDE.get(fertilizer, {}).get(
        language,
        f"Suivez les instructions sur l'emballage de {fertilizer}."
        if language == "fr"
        else f"Follow the instructions on the {fertilizer} package."
    )

    crops_text = (
        "\n".join([f"  {c['rank']}. {c['name']}" for c in crops])
        if crops else
        ("Aucune culture spécifique suggérée." if language == "fr"
         else "No specific crops suggested.")
    )

    issues_text = _build_issues_text(soil_health, language)

    # ── Build user message in the correct language ────────────────────────────
    if language == "fr":
        user_message = f"""
Nom de l'agriculteur : {farmer_name}

Lectures du capteur de sol :
  pH          : {readings['ph']}         (idéal : 6.0 – 7.0)
  Azote (N)   : {readings['nitrogen']} mg/kg   (idéal : 140 – 280)
  Phosphore(P): {readings['phosphorus']} mg/kg   (idéal : 10 – 40)
  Potassium(K): {readings['potassium']} mg/kg   (idéal : 120 – 280)
  Humidité    : {readings['humidity']}%        (idéal : 50 – 70%)
  Température : {readings['temperature']}°C         (idéal : 20 – 35°C)

{issues_text}

Engrais prédit par le modèle : {fertilizer}
Guide d'application : {guide}

Cultures suggérées :
{crops_text}

Rédige maintenant le rapport complet en FRANÇAIS pour {farmer_name}.
IMPORTANT : Réponds UNIQUEMENT en français. Pas un seul mot en anglais.
"""
    else:
        user_message = f"""
Farmer name: {farmer_name}

Soil sensor readings:
  pH          : {readings['ph']}         (ideal: 6.0 – 7.0)
  Nitrogen (N): {readings['nitrogen']} mg/kg   (ideal: 140 – 280)
  Phosphorus(P): {readings['phosphorus']} mg/kg  (ideal: 10 – 40)
  Potassium (K): {readings['potassium']} mg/kg  (ideal: 120 – 280)
  Humidity    : {readings['humidity']}%        (ideal: 50 – 70%)
  Temperature : {readings['temperature']}°C        (ideal: 20 – 35°C)

{issues_text}

Fertilizer predicted by the model: {fertilizer}
Application guide: {guide}

Suggested crops:
{crops_text}

Write the complete report in ENGLISH for {farmer_name}.
IMPORTANT: Respond ONLY in English. Not a single word in French.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_message}
            ],
            temperature=0.4,
            max_tokens=900
        )
        return response.choices[0].message.content

    except Exception as e:
        crops_fallback = ", ".join(c["name"] for c in crops) if crops else "N/A"
        if language == "fr":
            return (
                f"Engrais recommandé : {fertilizer}.\n"
                f"{guide}\n"
                f"Cultures suggérées : {crops_fallback}.\n\n"
                f"(Explication indisponible. Erreur : {str(e)[:80]})"
            )
        else:
            return (
                f"Recommended fertilizer: {fertilizer}.\n"
                f"{guide}\n"
                f"Suggested crops: {crops_fallback}.\n\n"
                f"(Explanation unavailable. Error: {str(e)[:80]})"
            )
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                *history   # full conversation — LLaMA sees everything
            ],
            temperature=0.5,
            max_tokens=300    # keep answers concise for mobile
        )
        return response.choices[0].message.content

    except Exception as e:
        return (
            "Désolé, je ne peux pas répondre maintenant. Réessayez dans un moment."
            if language == "fr"
            else "Sorry, I cannot answer right now. Please try again in a moment."
        )
