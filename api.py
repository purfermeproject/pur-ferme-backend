"""
api.py — Pur Ferme Backend API
Flask backend serving all data to the React frontend.
Zero hardcoded farm/soil data — everything is dynamic.

Endpoints:
  GET  /api/health
  GET  /api/geocode?location=Koraput
  GET  /api/weather?lat=18.8&lon=82.7
  GET  /api/climate?lat=18.8&lon=82.7
  POST /api/predict        (multipart image + location)
  POST /api/crop-plan      (JSON: location, sowing_date, lat, lon)
"""

import os, sys, pathlib, platform, warnings, requests, json
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# ── Cross-platform FastAI fix ─────────────────────────────────────────────────
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath
try:
    import plum
    if not hasattr(plum, "function"):
        sys.modules["plum.function"] = plum
except ImportError:
    pass

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from io import BytesIO
from fastai.vision.all import load_learner, CrossEntropyLossFlat

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE      = os.environ.get("MODEL_PATH", os.path.join(BASE_DIR, "pur_ferme_v4_clean.pkl"))
API_KEY_WEATHER = os.getenv("OPENWEATHER_API_KEY")
GEMINI_KEY      = os.getenv("GEMINI_API_KEY")
OPENAI_KEY      = os.getenv("OPENAI_API_KEY")

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Load Model ────────────────────────────────────────────────────────────────
print("🔄 Loading model...")
try:
    if platform.system() == "Windows":
        pathlib.PosixPath = pathlib.WindowsPath
    learn           = load_learner(MODEL_FILE, cpu=True)
    learn.loss_func = CrossEntropyLossFlat()
    MODEL           = learn.model
    VOCAB           = learn.dls.vocab
    MODEL.eval()
    print(f"✅ Model loaded. Classes: {list(VOCAB)}")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    MODEL = None
    VOCAB = []

# ── Image transform ───────────────────────────────────────────────────────────
_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Grade rules ───────────────────────────────────────────────────────────────
GRADE_RULES = {
    "Healthy": 1,
    "Water_Stress": 2, "Nitrogen_Deficiency": 2, "Phosphorus_Deficiency": 2,
    "Rust": 3, "Blast": 3, "Downy_Mildew": 3,
}

def get_grade(pred_class, humidity):
    base    = GRADE_RULES.get(pred_class, 2)
    penalty = 1 if (humidity and humidity > 80) else 0
    return min(base + penalty, 3)

# ═════════════════════════════════════════════════════════════════════════════
#  LLM
# ═════════════════════════════════════════════════════════════════════════════
# def call_llm(prompt: str) -> str:
#     if GEMINI_KEY:
#         import google.generativeai as genai
#         genai.configure(api_key=GEMINI_KEY)
#         # m = genai.GenerativeModel("gemini-2.5-flash")
#         # m = genai.GenerativeModel("gemini-1.5-flash")
#         # m = genai.GenerativeModel("gemini-2.0-flash")
#         m = genai.GenerativeModel("gemini-3-flash")
#         return m.generate_content(prompt).text
#     elif OPENAI_KEY:
#         from openai import OpenAI
#         client = OpenAI(api_key=OPENAI_KEY)
#         resp   = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=2000,
#         )
#         return resp.choices[0].message.content
#     return ""

def call_llm(prompt: str) -> str:
    # Try Groq first (free, generous limits)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        from groq import Groq
        client   = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        return response.choices[0].message.content

    # Fallback to Gemini
    if GEMINI_KEY:
        from google import genai
        client   = genai.Client(api_key=GEMINI_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text

    return ""

# LLM_NAME = "Gemini 2.5 Flash" if GEMINI_KEY else ("GPT-4o Mini" if OPENAI_KEY else "None")
LLM_NAME = "Llama 3.3 70B (Groq)" if os.getenv("GROQ_API_KEY") else ("Gemini 2.5 Flash" if GEMINI_KEY else "None")
# ═════════════════════════════════════════════════════════════════════════════
#  EXTERNAL DATA HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def geocode_location(location_name: str):
    """Nominatim: location name → lat, lon, display_name"""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": f"{location_name}, India", "format": "json", "limit": 1},
            headers={"User-Agent": "PurFerme-API/2.0"},
            timeout=10,
        )
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"]), data[0]["display_name"]
    except Exception:
        pass
    return None, None, None


def get_soil_from_llm(location_name: str, lat: float, lon: float) -> dict:
    """Ask Gemini for soil data — works for ANY location, no hardcoding."""
    prompt = f"""For the agricultural location "{location_name}" (coordinates: {lat:.4f}N, {lon:.4f}E) in India:
Provide the typical soil information for this region.
Return ONLY a valid JSON object with exactly these keys, nothing else:
{{
    "type": "soil type name (e.g. Red Laterite, Black Cotton Soil)",
    "ph": "typical pH range (e.g. 5.5-6.5)",
    "deficiencies": "main nutrient deficiencies (e.g. Low Phosphorus, Low Zinc)"
}}
No explanation, no markdown, no code blocks. Just the raw JSON."""
    try:
        result = call_llm(prompt)
        # Clean up response
        result = result.strip()
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        return json.loads(result.strip())
    except Exception:
        return {
            "type":         "Mixed Agricultural Soil",
            "ph":           "6.0–7.0",
            "deficiencies": "Soil test recommended for accurate data",
        }


def get_weather(lat: float, lon: float) -> dict:
    """OpenWeatherMap live weather for any coordinates."""
    if not API_KEY_WEATHER:
        return {"success": False, "error": "No weather API key"}
    try:
        resp = requests.get(
            "http://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": API_KEY_WEATHER, "units": "metric"},
            timeout=10,
        )
        resp.raise_for_status()
        d = resp.json()
        return {
            "success":    True,
            "temp":       round(d["main"]["temp"], 1),
            "humidity":   d["main"]["humidity"],
            "condition":  d["weather"][0]["main"],
            "wind_speed": round(d["wind"]["speed"], 1),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_climate_data(lat: float, lon: float) -> dict:
    """Open-Meteo: last 30 days + 10-year monthly averages. Free, no key."""
    BASE  = "https://archive-api.open-meteo.com/v1/archive"
    DAILY = "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum,wind_speed_10m_mean"
    today = datetime.today()
    result = {"elevation": None, "recent_30d": None, "monthly_avg": None}

    # Last 30 days
    try:
        end   = (today - timedelta(days=3)).strftime("%Y-%m-%d")
        start = (today - timedelta(days=33)).strftime("%Y-%m-%d")
        resp  = requests.get(BASE, params={
            "latitude": lat, "longitude": lon,
            "start_date": start, "end_date": end,
            "daily": DAILY, "timezone": "Asia/Kolkata",
        }, timeout=20)
        resp.raise_for_status()
        d  = resp.json()
        result["elevation"] = d.get("elevation")
        dd = d.get("daily", {})
        temps = [t for t in (dd.get("temperature_2m_mean")       or []) if t is not None]
        hums  = [h for h in (dd.get("relative_humidity_2m_mean") or []) if h is not None]
        rains = [r for r in (dd.get("precipitation_sum")         or []) if r is not None]
        winds = [w for w in (dd.get("wind_speed_10m_mean")       or []) if w is not None]
        if temps:
            mid = len(temps) // 2
            result["recent_30d"] = {
                "avg_temp":     round(sum(temps) / len(temps), 1),
                "avg_humidity": round(sum(hums)  / len(hums),  1) if hums  else None,
                "total_rain":   round(sum(rains), 1)               if rains else None,
                "avg_wind":     round(sum(winds) / len(winds), 1)  if winds else None,
                "temp_trend":   "rising" if sum(temps[mid:]) > sum(temps[:mid]) else "falling",
            }
    except Exception:
        pass

    # 10-year monthly averages
    try:
        resp = requests.get(BASE, params={
            "latitude": lat, "longitude": lon,
            "start_date": "2015-01-01", "end_date": "2024-12-31",
            "daily": "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum",
            "timezone": "Asia/Kolkata",
        }, timeout=40)
        resp.raise_for_status()
        d  = resp.json()
        dd = d.get("daily", {})
        if result["elevation"] is None:
            result["elevation"] = d.get("elevation")
        dates = dd.get("time", [])
        temps = dd.get("temperature_2m_mean",        [])
        hums  = dd.get("relative_humidity_2m_mean",  [])
        rains = dd.get("precipitation_sum",          [])
        m_t, m_h, m_r = defaultdict(list), defaultdict(list), defaultdict(list)
        for dt, t, h, r in zip(dates, temps, hums, rains):
            mo = int(dt[5:7])
            if t is not None: m_t[mo].append(t)
            if h is not None: m_h[mo].append(h)
            if r is not None: m_r[mo].append(r)
        names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        result["monthly_avg"] = {
            mo: {
                "name":     names[mo - 1],
                "temp":     round(sum(m_t[mo]) / len(m_t[mo]), 1) if m_t[mo] else None,
                "humidity": round(sum(m_h[mo]) / len(m_h[mo]), 1) if m_h[mo] else None,
                "rain":     round(sum(m_r[mo]) / len(m_r[mo]), 1) if m_r[mo] else None,
            }
            for mo in range(1, 13)
        }
    except Exception:
        pass

    return result

# ═════════════════════════════════════════════════════════════════════════════
#  PROMPT BUILDERS
# ═════════════════════════════════════════════════════════════════════════════

def _climate_summary(climate: dict, location_name: str) -> str:
    now = datetime.now()
    r30 = climate.get("recent_30d") or {}
    ma  = climate.get("monthly_avg") or {}
    curr_mo = now.month

    recent = (
        f"LAST 30 DAYS FOR {location_name.upper()}:\n"
        f"  Avg Temp: {r30.get('avg_temp','N/A')}°C ({r30.get('temp_trend','unknown')} trend) | "
        f"Avg Humidity: {r30.get('avg_humidity','N/A')}% | "
        f"Total Rain: {r30.get('total_rain','N/A')}mm"
    ) if r30.get("avg_temp") else "LAST 30 DAYS: Unavailable"

    hist_rows = ""
    deviation = ""
    if ma:
        names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        hist_rows = "\n".join(
            f"  {ma[m]['name']:>3} | {str(ma[m]['temp'])+'°C':>7} | "
            f"{str(ma[m]['humidity'])+'%':>7} | {str(ma[m]['rain'])+'mm':>8} | "
            f"{'🔴 VERY HIGH' if (ma[m]['humidity'] or 0) >= 80 else '🟠 HIGH' if (ma[m]['humidity'] or 0) >= 70 else '🟡 MOD' if (ma[m]['humidity'] or 0) >= 60 else '🟢 LOW'}"
            for m in range(1, 13) if ma.get(m) and ma[m].get("temp")
        )
        curr = ma.get(curr_mo, {})
        if curr.get("temp") and r30.get("avg_temp"):
            td = round(r30["avg_temp"] - curr["temp"], 1)
            hd = round(r30["avg_humidity"] - (curr.get("humidity") or 0), 1)
            deviation = (
                f"\nTHIS MONTH vs 10-YR AVERAGE:\n"
                f"  Temp: {'+' if td>0 else ''}{td}°C {'WARMER' if td>0 else 'COOLER'} | "
                f"Humidity: {'+' if hd>0 else ''}{hd}% {'MORE HUMID' if hd>0 else 'DRIER'} than normal"
            )

    return f"""{recent}

10-YEAR MONTHLY AVERAGES (2015-2024):
  Month | Avg Temp | Avg Hum  |  Avg Rain | Disease Risk
{hist_rows}
{deviation}"""


def build_image_prompt(pred_class, confidence, grade, probs, vocab,
                        weather, climate, location_name, soil, elevation) -> str:
    now        = datetime.now()
    grade_desc = {1:"Cleared for Supply Chain", 2:"Conditional Acceptance", 3:"QUARANTINE"}[grade]
    prob_lines = "\n".join(f"  {cls}: {float(p)*100:.1f}%" for cls, p in zip(vocab, probs))
    wx = (
        f"  Temp: {weather['temp']}°C | Humidity: {weather['humidity']}% | "
        f"Wind: {weather.get('wind_speed','N/A')}m/s | {weather['condition']}"
    ) if weather and weather.get("success") else "  Unavailable"

    return f"""You are a sharp field agronomist with deep knowledge of Indian agriculture.
Use the real measured data below. No generic advice. Every point must reference actual numbers.

LOCATION: {location_name} | Elevation: {f'{elevation:.0f}m' if elevation else 'N/A'}
SOIL: {soil['type']} | {soil['ph']} | Deficiencies: {soil['deficiencies']}
DATE: {now.strftime('%d %B %Y, %H:%M')} IST

AI MODEL (FastAI ResNet-18):
  Detected: {pred_class} | Confidence: {confidence:.1f}% | Grade: {grade} — {grade_desc}
  All probabilities:
{prob_lines}

LIVE WEATHER:
{wx}

{_climate_summary(climate, location_name)}

RULES: Reference actual numbers. No vague advice. Exact products/doses/timings.

## 🔍 What Is Happening & Why
## 🌦️ How This Season Compares to Normal
## ⚡ Immediate Actions — Next 48 Hours
## 📅 7-Day Plan
## 🔮 Next 30 Days Based on Historical Pattern
## 🚛 Grade {grade} — Commercial Impact & Actions
"""


def build_crop_plan_prompt(location_name, sowing_date, climate, soil, elevation) -> str:
    now     = datetime.now()
    ma      = climate.get("monthly_avg") or {}
    r30     = climate.get("recent_30d")  or {}
    sow_dt  = datetime.strptime(str(sowing_date), "%Y-%m-%d")
    harv_dt = sow_dt + timedelta(days=85)

    month_rows = ""
    if ma:
        for m in range(1, 13):
            d   = ma.get(m, {})
            hv  = d.get("humidity") or 0
            risk = ("🔴 VERY HIGH" if hv >= 80 else "🟠 HIGH" if hv >= 70
                    else "🟡 MODERATE" if hv >= 60 else "🟢 LOW")
            month_rows += (
                f"  {d.get('name',''):>3} | {str(d.get('temp'))+'°C' if d.get('temp') else 'N/A':>7} | "
                f"{str(d.get('humidity'))+'%' if d.get('humidity') else 'N/A':>7} | "
                f"{str(d.get('rain'))+'mm' if d.get('rain') else 'N/A':>8} | {risk}\n"
            )

    curr_mo   = now.month
    curr_hist = ma.get(curr_mo, {})
    season_ctx = ""
    if curr_hist.get("temp") and r30.get("avg_temp"):
        td = round(r30["avg_temp"] - curr_hist["temp"], 1)
        hd = round(r30["avg_humidity"] - (curr_hist.get("humidity") or 0), 1)
        season_ctx = (
            f"\nCURRENT SEASON vs NORMAL: "
            f"{'+' if td>0 else ''}{td}°C {'warmer' if td>0 else 'cooler'} and "
            f"{'+' if hd>0 else ''}{hd}% {'more humid' if hd>0 else 'drier'} than 10yr average."
        )

    return f"""Build a complete foxtail millet season plan using ONLY the actual data below.

LOCATION: {location_name} | Elevation: {f'{elevation:.0f}m' if elevation else 'N/A'}
SOIL: {soil['type']} | {soil['ph']} | Deficiencies: {soil['deficiencies']}
CROP: Foxtail Millet | SOWING: {sowing_date} | EST. HARVEST: {harv_dt.strftime('%d %B %Y')}

10-YEAR MONTHLY CLIMATE DATA FOR {location_name.upper()}:
  Month | Avg Temp | Avg Hum  |  Avg Rain | Disease Risk
{month_rows}
{season_ctx}

## 📍 Location Climate Profile & Key Risk Windows
## 🗓️ Stage-by-Stage Plan with Exact Dates (from {sowing_date} to {harv_dt.strftime('%d %b %Y')})
## 🌿 Fertilizer Schedule (specific to {soil['type']} with {soil['deficiencies']})
## 💊 Disease Management Calendar (based on historical humidity data above)
## 💧 Irrigation Schedule (based on historical rainfall gaps)
## ⚠️ This Season's Specific Risks
## 🌾 Harvest & Post-Harvest Guide
"""

# ═════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":  "ok",
        "model":   "pur_ferme_v4_clean" if MODEL else "not loaded",
        "classes": list(VOCAB) if VOCAB else [],
        "llm":     LLM_NAME,
    })


@app.route("/api/geocode", methods=["GET"])
def geocode():
    location = request.args.get("location", "").strip()
    if not location:
        return jsonify({"success": False, "error": "location param required"}), 400

    lat, lon, display_name = geocode_location(location)
    if not lat:
        return jsonify({"success": False, "error": f"Could not find '{location}'"}), 404

    # Get soil from Gemini — no hardcoding
    soil = get_soil_from_llm(location, lat, lon)

    return jsonify({
        "success":      True,
        "lat":          lat,
        "lon":          lon,
        "display_name": display_name,
        "soil":         soil,
    })


@app.route("/api/weather", methods=["GET"])
def weather():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "lat and lon required"}), 400
    return jsonify(get_weather(lat, lon))


@app.route("/api/climate", methods=["GET"])
def climate():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"success": False, "error": "lat and lon required"}), 400
    return jsonify(get_climate_data(lat, lon))


@app.route("/api/predict", methods=["POST"])
def predict():
    if not MODEL:
        return jsonify({"success": False, "error": "Model not loaded"}), 500
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image uploaded"}), 400

    # Get location data from form
    location_name = request.form.get("location", "Unknown Location")
    lat  = request.form.get("lat",  type=float)
    lon  = request.form.get("lon",  type=float)
    soil = json.loads(request.form.get("soil", "{}"))

    try:
        # Run model
        pil_img = Image.open(request.files["image"].stream).convert("RGB")
        tensor  = _transform(pil_img).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(MODEL(tensor)[0], dim=0)
        idx        = probs.argmax().item()
        pred_class = str(VOCAB[idx])
        confidence = float(probs[idx]) * 100
        all_probs  = {str(VOCAB[i]): round(float(probs[i]) * 100, 2) for i in range(len(VOCAB))}

        # Get weather + climate
        wx      = get_weather(lat, lon) if lat and lon else {"success": False}
        climate = get_climate_data(lat, lon) if lat and lon else {}
        elev    = climate.get("elevation")

        # Grade
        humidity = wx.get("humidity") if wx.get("success") else None
        grade    = get_grade(pred_class, humidity)
        grade_labels = {
            1: "Grade 1 — Cleared for Supply Chain",
            2: "Grade 2 — Conditional Acceptance",
            3: "Grade 3 — Rejected / Quarantine",
        }

        # Gemini report
        llm_report = ""
        # if GEMINI_KEY or OPENAI_KEY:
        if GEMINI_KEY or OPENAI_KEY or os.getenv("GROQ_API_KEY"):
            try:
                # llm_report = call_llm(build_image_prompt(
                  llm_report = call_llm(build_image_prompt(
                    pred_class, confidence, grade,
                    probs, VOCAB, wx, climate,
                    location_name, soil, elev
                ))
            except Exception as e:
                llm_report = f"Report generation failed: {e}"

        return jsonify({
            "success":    True,
            "pred_class": pred_class,
            "confidence": round(confidence, 2),
            "all_probs":  all_probs,
            "grade":      grade,
            "grade_label": grade_labels[grade],
            "weather":    wx,
            "elevation":  elev,
            "llm_report": llm_report,
            "llm_name":   LLM_NAME,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/crop-plan", methods=["POST"])
def crop_plan():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "JSON body required"}), 400

    location_name = data.get("location", "Unknown")
    sowing_date   = data.get("sowing_date")
    lat           = data.get("lat")
    lon           = data.get("lon")
    soil          = data.get("soil", {})

    if not all([sowing_date, lat, lon]):
        return jsonify({"success": False, "error": "location, sowing_date, lat, lon required"}), 400

    if not (GEMINI_KEY or OPENAI_KEY):
        return jsonify({"success": False, "error": "No LLM key configured"}), 500

    try:
        climate  = get_climate_data(lat, lon)
        elev     = climate.get("elevation")
        sow_dt   = datetime.strptime(sowing_date, "%Y-%m-%d")
        harv_dt  = sow_dt + timedelta(days=85)
        plan     = call_llm(build_crop_plan_prompt(
            location_name, sowing_date, climate, soil, elev
        ))

        return jsonify({
            "success":       True,
            "plan":          plan,
            "harvest_est":   harv_dt.strftime("%d %B %Y"),
            "total_days":    85,
            "llm_name":      LLM_NAME,
            "climate_used":  True,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/generate-pdf", methods=["POST"])
def generate_pdf():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    import io, re

    data         = request.get_json()
    content      = data.get("content", "")
    filename     = data.get("filename", "report")
    location     = data.get("location", "")
    result       = data.get("result", {})
    report_type  = data.get("type", "analysis")
    sowing_date  = data.get("sowing_date", "")
    now          = datetime.now().strftime("%d %B %Y, %H:%M IST")

    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=20*mm, leftMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "Title", parent=styles["Normal"],
        fontSize=18, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#14532d"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=11, textColor=colors.HexColor("#6b7280"),
        spaceAfter=12,
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Normal"],
        fontSize=13, fontName="Helvetica-Bold",
        textColor=colors.HexColor("#14532d"),
        spaceBefore=14, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=16,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "Bullet", parent=styles["Normal"],
        fontSize=10, leading=16,
        leftIndent=16,
        textColor=colors.HexColor("#374151"),
        spaceAfter=3,
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("🌾 PUR FERME TRACEABILITY SYSTEM", title_style))
    if report_type == "analysis":
        story.append(Paragraph("Crop Health Certificate of Analysis", subtitle_style))
    else:
        story.append(Paragraph("Foxtail Millet Season Plan", subtitle_style))
    story.append(Paragraph(f"Location: {location}  |  Generated: {now}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#16a34a"), spaceAfter=12))

    # ── Analysis specific header ───────────────────────────────────────────────
    if report_type == "analysis" and result:
        grade    = result.get("grade", "")
        g_colors = {1: "#16a34a", 2: "#d97706", 3: "#dc2626"}
        g_color  = g_colors.get(grade, "#374151")

        table_data = [
            ["Detected Condition", "AI Confidence", "Supply Chain Grade"],
            [
                result.get("pred_class", "N/A"),
                f"{result.get('confidence', 0):.1f}%",
                f"Grade {grade} — {result.get('grade_label','').split('—')[-1].strip() if result.get('grade_label') else ''}",
            ],
        ]
        t = Table(table_data, colWidths=[55*mm, 55*mm, 65*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#f0fdf4")),
            ("TEXTCOLOR",    (0,0), (-1,0), colors.HexColor("#14532d")),
            ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",     (0,0), (-1,0), 10),
            ("FONTNAME",     (0,1), (-1,1), "Helvetica-Bold"),
            ("FONTSIZE",     (0,1), (-1,1), 12),
            ("TEXTCOLOR",    (2,1), (2,1),  colors.HexColor(g_color)),
            ("ALIGN",        (0,0), (-1,-1), "CENTER"),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white]),
            ("BOX",          (0,0), (-1,-1), 1, colors.HexColor("#e5e7eb")),
            ("INNERGRID",    (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
            ("TOPPADDING",   (0,0), (-1,-1), 8),
            ("BOTTOMPADDING",(0,0), (-1,-1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

    # ── Season plan header ─────────────────────────────────────────────────────
    if report_type == "plan" and result:
        story.append(Paragraph(
            f"Sowing Date: {sowing_date}  |  Estimated Harvest: {result.get('harvest_est','')}  |  Duration: 85 days",
            body_style
        ))
        story.append(Spacer(1, 8))

    # ── Main LLM content ──────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb"), spaceAfter=10))
    story.append(Paragraph("AI Agronomist Report", h2_style))

    # Parse markdown → PDF paragraphs
    for line in content.split("\n"):
        s = line.strip()
        if not s:
            story.append(Spacer(1, 4))
        elif s.startswith("## "):
            story.append(Paragraph(s[3:], h2_style))
        elif s.startswith("# "):
            story.append(Paragraph(s[2:], h2_style))
        elif s.startswith(("- ", "* ")):
            clean = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s[2:])
            story.append(Paragraph(f"• {clean}", bullet_style))
        elif re.match(r"^\d+\.\s", s):
            clean = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", re.sub(r"^\d+\.\s", "", s))
            story.append(Paragraph(f"• {clean}", bullet_style))
        else:
            clean = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
            story.append(Paragraph(clean, body_style))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e5e7eb")))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Pur Ferme Traceability System  |  AI: {LLM_NAME}  |  Advisory only — validate with local agronomist",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=8,
                       textColor=colors.HexColor("#9ca3af"), alignment=TA_CENTER)
    ))

    doc.build(story)
    buffer.seek(0)

    from flask import send_file
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{filename}.pdf",
    )

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"🌾 Pur Ferme API starting...")
    print(f"   LLM     : {LLM_NAME}")
    print(f"   Model   : {MODEL_FILE}")
    print(f"   Endpoints:")
    print(f"   GET  /api/health")
    print(f"   GET  /api/geocode?location=Koraput")
    print(f"   GET  /api/weather?lat=18.8&lon=82.7")
    print(f"   GET  /api/climate?lat=18.8&lon=82.7")
    print(f"   POST /api/predict")
    print(f"   POST /api/crop-plan")
    app.run(host="0.0.0.0", port=5000, debug=False)