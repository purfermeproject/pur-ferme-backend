"""
weather_service.py
Fetches live weather for Koraput using exact coordinates.
Drop this file next to app.py — no path changes needed.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()


def get_koraput_weather(api_key: str) -> dict:
    """
    Fetch current weather for Koraput, Odisha using GPS coordinates.

    Returns dict with keys:
        success    : bool
        temp       : float  (°C)
        humidity   : int    (%)
        condition  : str    (e.g. "Rain", "Clear")
        wind_speed : float  (m/s)
        error      : str    (only present if success=False)
    """
    lat, lon = 18.8115, 82.7121
    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "success":    True,
            "temp":       data["main"]["temp"],
            "humidity":   data["main"]["humidity"],
            "condition":  data["weather"][0]["main"],
            "wind_speed": data["wind"]["speed"],
        }
    except Exception as e:
        return {"success": False, "error": f"❌ Connection Error: {e}"}


# ── Quick test — run this file directly to verify your API key ────────────────
if __name__ == "__main__":
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    if not API_KEY:
        print("❌ OPENWEATHER_API_KEY not found in .env file!")
    else:
        print("☁️  Fetching live weather for Koraput…")
        result = get_koraput_weather(API_KEY)
        if result["success"]:
            print(f"✅ Success!")
            print(f"   Temperature : {result['temp']}°C")
            print(f"   Humidity    : {result['humidity']}%")
            print(f"   Wind Speed  : {result['wind_speed']} m/s")
            print(f"   Condition   : {result['condition']}")
        else:
            print(result["error"])