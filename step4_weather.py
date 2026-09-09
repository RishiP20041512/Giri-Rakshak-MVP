import requests
import pandas as pd


# =========================================================
# STEP 4 — REAL WEATHER FROM LAT/LON
# =========================================================

LAT = 26.1445
LON = 91.7362

URL = "https://api.open-meteo.com/v1/forecast"


def get_weather(lat, lon):

    params = {
        "latitude": lat,
        "longitude": lon,

        # Current weather
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "surface_pressure,"
            "wind_speed_10m"
        ),

        # Hourly forecast
        "hourly": (
            "temperature_2m,"
            "precipitation,"
            "precipitation_probability,"
            "relative_humidity_2m,"
            "surface_pressure,"
            "wind_speed_10m"
        ),

        # 7-day forecast
        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum,"
            "precipitation_probability_max"
        ),

        "timezone": "auto",

        # Previous 3 days + next 7 days
        "past_days": 3,
        "forecast_days": 7,
    }

    print("\nConnecting to weather service...")

    response = requests.get(
        URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def main():

    print("=" * 65)
    print("GIRI-RAKSHAK — STEP 4")
    print("REAL WEATHER FROM LATITUDE + LONGITUDE")
    print("=" * 65)

    print(f"\nLatitude  : {LAT}")
    print(f"Longitude : {LON}")

    # -----------------------------------------------------
    # GET WEATHER
    # -----------------------------------------------------

    try:

        data = get_weather(
            LAT,
            LON
        )

    except Exception as e:

        print("\n❌ Weather request failed:")
        print(e)

        return

    print("\n✅ Weather data received")

    # -----------------------------------------------------
    # CURRENT WEATHER
    # -----------------------------------------------------

    current = data["current"]

    print("\n" + "=" * 65)
    print("CURRENT WEATHER")
    print("=" * 65)

    print(
        f"\nTime              : "
        f"{current['time']}"
    )

    print(
        f"Temperature       : "
        f"{current['temperature_2m']} °C"
    )

    print(
        f"Humidity          : "
        f"{current['relative_humidity_2m']} %"
    )

    print(
        f"Precipitation     : "
        f"{current['precipitation']} mm"
    )

    print(
        f"Surface pressure  : "
        f"{current['surface_pressure']} hPa"
    )

    print(
        f"Wind speed        : "
        f"{current['wind_speed_10m']} km/h"
    )

    # -----------------------------------------------------
    # HOURLY DATA
    # -----------------------------------------------------

    hourly = data["hourly"]

    hourly_df = pd.DataFrame({
        "time": hourly["time"],
        "temperature_c": hourly["temperature_2m"],
        "rainfall_mm": hourly["precipitation"],
        "rain_probability": hourly[
            "precipitation_probability"
        ],
        "humidity_percent": hourly[
            "relative_humidity_2m"
        ],
        "pressure_hpa": hourly[
            "surface_pressure"
        ],
        "wind_kmh": hourly[
            "wind_speed_10m"
        ],
    })

    hourly_df["time"] = pd.to_datetime(
        hourly_df["time"]
    )

    # -----------------------------------------------------
    # NEXT 24 HOURS
    # -----------------------------------------------------

    future_24h = hourly_df[
        hourly_df["time"] >= pd.Timestamp.now(
            tz=hourly_df["time"].dt.tz
        )
    ].head(24)

    rainfall_24h = future_24h[
        "rainfall_mm"
    ].sum()

    # -----------------------------------------------------
    # NEXT 72 HOURS
    # -----------------------------------------------------

    future_72h = hourly_df[
        hourly_df["time"] >= pd.Timestamp.now(
            tz=hourly_df["time"].dt.tz
        )
    ].head(72)

    rainfall_72h = future_72h[
        "rainfall_mm"
    ].sum()

    max_probability_24h = future_24h[
        "rain_probability"
    ].max()

    print("\n" + "=" * 65)
    print("RAINFALL FORECAST")
    print("=" * 65)

    print(
        f"\nForecast rainfall next 24h : "
        f"{rainfall_24h:.2f} mm"
    )

    print(
        f"Forecast rainfall next 72h : "
        f"{rainfall_72h:.2f} mm"
    )

    print(
        f"Maximum rain probability "
        f"next 24h              : "
        f"{max_probability_24h:.0f}%"
    )

    # -----------------------------------------------------
    # 24-HOUR HOURLY TABLE
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("NEXT 24 HOURS")
    print("=" * 65)

    print(
        future_24h[
            [
                "time",
                "temperature_c",
                "rainfall_mm",
                "rain_probability",
            ]
        ].to_string(index=False)
    )

    # -----------------------------------------------------
    # DAILY FORECAST
    # -----------------------------------------------------

    daily = data["daily"]

    daily_df = pd.DataFrame({
        "date": daily["time"],
        "max_temp_c": daily[
            "temperature_2m_max"
        ],
        "min_temp_c": daily[
            "temperature_2m_min"
        ],
        "rainfall_mm": daily[
            "precipitation_sum"
        ],
        "rain_probability": daily[
            "precipitation_probability_max"
        ],
    })

    print("\n" + "=" * 65)
    print("7-DAY FORECAST")
    print("=" * 65)

    print(
        daily_df.to_string(index=False)
    )

    print("\n" + "=" * 65)
    print("STEP 4 COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()