"""Weather Data Architecture and Provider Interfaces.

Establishes provider interfaces to ingest external weather API data
(e.g., Open-Meteo, NASA POWER, or Sri Lanka Department of Meteorology)
without requiring physical on-farm IoT sensors or weather hardware.

Distinguishes between:
- Farmer-reported observations (CROP_HEALTH, IRRIGATION)
- External API-derived data (WEATHER_API / API_DERIVED)
- Historical dataset baseline (HISTORICAL_DATASET)
- Synthetic / simulated experimental data (SYNTHETIC_SIMULATED)
"""

import json
import ssl
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from datetime import date as dt_date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config import (
    DATA_SOURCE_API,
    DATA_SOURCE_SYNTHETIC,
    DISTRICT_COORDINATES,
    DISTRICTS,
)


class WeatherAPIError(Exception):
    """Exception raised when weather data cannot be fetched or parsed from external API."""

    pass


class AbstractWeatherProvider(ABC):
    """Abstract Base Class for external weather data providers."""

    @abstractmethod
    def get_weather_for_date(
        self, district: str, date: str
    ) -> Dict[str, Any]:
        """Fetch daily/weekly environmental parameters for a given district and date."""
        pass

    @abstractmethod
    def get_weather_range(
        self, district: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """Fetch historical weather series over a date range."""
        pass


class MockWeatherProvider(AbstractWeatherProvider):
    """Offline, deterministic weather simulator for testing and validation.

    Explicitly tagged as SYNTHETIC_SIMULATED.
    """

    # Realistic Sri Lankan Ginger District Baselines (Wet/Intermediate Zones)
    DISTRICT_CLIMATOLOGY = {
        "Kandy": {"temp": 24.5, "rain": 45.0, "rh": 80.0, "solar": 18.0},
        "Kurunegala": {"temp": 28.0, "rain": 35.0, "rh": 75.0, "solar": 20.0},
        "Matale": {"temp": 25.5, "rain": 40.0, "rh": 78.0, "solar": 19.0},
        "Badulla": {"temp": 23.0, "rain": 38.0, "rh": 82.0, "solar": 17.5},
        "Kegalle": {"temp": 27.0, "rain": 55.0, "rh": 85.0, "solar": 18.5},
        "Galle": {"temp": 27.5, "rain": 60.0, "rh": 84.0, "solar": 18.0},
        "Matara": {"temp": 28.0, "rain": 50.0, "rh": 82.0, "solar": 19.0},
        "Ratnapura": {"temp": 27.0, "rain": 65.0, "rh": 86.0, "solar": 17.5},
        "Monaragala": {"temp": 28.5, "rain": 30.0, "rh": 72.0, "solar": 21.0},
        "Nuwara Eliya": {"temp": 16.5, "rain": 48.0, "rh": 88.0, "solar": 16.0},
    }

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def get_weather_for_date(
        self, district: str, date: str
    ) -> Dict[str, Any]:
        """Generate deterministic simulated weather observation for given district/date."""
        clim = self.DISTRICT_CLIMATOLOGY.get(district, {"temp": 26.0, "rain": 45.0, "rh": 80.0, "solar": 18.5})

        # Deterministic day-of-year variation
        dt = datetime.strptime(str(date)[:10], "%Y-%m-%d")
        doy = dt.timetuple().tm_yday
        seasonal_temp = clim["temp"] + 2.0 * np.sin(2 * np.pi * doy / 365.0)
        seasonal_rain = max(0.0, clim["rain"] + 15.0 * np.cos(2 * np.pi * doy / 365.0))

        return {
            "Date": str(date)[:10],
            "District": district,
            "Weekly_Rainfall_mm": round(float(seasonal_rain), 2),
            "Avg_Temperature_C": round(float(seasonal_temp), 2),
            "Relative_Humidity_pct": round(float(clim["rh"]), 2),
            "Solar_Radiation_MJ_m2_day": round(float(clim["solar"]), 2),
            "Data_Source": DATA_SOURCE_SYNTHETIC,
            "Source": "Mock-Simulator",
        }

    def get_weather_range(
        self, district: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """Generate time-series range of simulated weather observations."""
        start_dt = datetime.strptime(str(start_date)[:10], "%Y-%m-%d")
        end_dt = datetime.strptime(str(end_date)[:10], "%Y-%m-%d")

        records = []
        cur_dt = start_dt
        while cur_dt <= end_dt:
            records.append(self.get_weather_for_date(district, cur_dt.strftime("%Y-%m-%d")))
            cur_dt += timedelta(days=7)  # Weekly observations

        return records


class OpenMeteoWeatherProvider(AbstractWeatherProvider):
    """External Open-Meteo Weather API Integration.

    Retrieves 7-day historical precipitation and meteorological parameters
    for a given Sri Lankan district and observation date.
    Strictly excludes future dates to prevent temporal data leakage.
    """

    def __init__(
        self,
        archive_url: str = "https://archive-api.open-meteo.com/v1/archive",
        forecast_url: str = "https://api.open-meteo.com/v1/forecast",
        timeout: float = 10.0,
    ):
        self.archive_url = archive_url
        self.forecast_url = forecast_url
        self.timeout = timeout
        self._cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._ssl_ctx = self._create_ssl_context()

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create a robust SSL context using certifi if available."""
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            try:
                return ssl.create_default_context()
            except Exception:
                return ssl._create_unverified_context()

    def _get_district_coordinates(self, district: str) -> Tuple[float, float]:
        """Resolve geographical coordinates for canonical district."""
        if district in DISTRICT_COORDINATES:
            return DISTRICT_COORDINATES[district]
        # Case-insensitive fallback lookup
        for d, coords in DISTRICT_COORDINATES.items():
            if d.lower() == district.strip().lower():
                return coords
        raise WeatherAPIError(
            f"Unknown or unsupported Sri Lankan district '{district}'. "
            f"Supported districts: {list(DISTRICT_COORDINATES.keys())}"
        )

    def _fetch_from_endpoint(
        self, endpoint: str, lat: float, lon: float, start_date: str, end_date: str
    ) -> Optional[Dict[str, Any]]:
        """Perform HTTP request against Open-Meteo endpoint."""
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "precipitation_sum,temperature_2m_mean,relative_humidity_2m_mean,shortwave_radiation_sum",
            "timezone": "auto",
        }
        url = f"{endpoint}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "GingerYieldAI-Research/1.0"})

        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl_ctx) as resp:
                if resp.status != 200:
                    return None
                data = json.loads(resp.read().decode("utf-8"))
                if "daily" in data and isinstance(data["daily"], dict):
                    return data["daily"]
        except Exception as e:
            # Fallback to unverified SSL if default verification failed
            try:
                unverified_ctx = ssl._create_unverified_context()
                with urllib.request.urlopen(req, timeout=self.timeout, context=unverified_ctx) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        if "daily" in data and isinstance(data["daily"], dict):
                            return data["daily"]
            except Exception:
                pass
            return None

        return None

    def get_weather_for_date(
        self, district: str, date: str
    ) -> Dict[str, Any]:
        """Fetch 7-day historical precipitation and weather ending on given date.

        Parameters
        ----------
        district : str
            Canonical Sri Lankan district name.
        date : str or datetime or date
            Observation date T (YYYY-MM-DD).

        Returns
        -------
        Dict[str, Any]
            Dictionary containing 7-day cumulative rainfall and average weather parameters.

        Raises
        ------
        WeatherAPIError
            If network fails, response is invalid, or district is unsupported.
        """
        # Parse observation date T
        if isinstance(date, str):
            obs_dt = datetime.strptime(str(date)[:10], "%Y-%m-%d").date()
        elif isinstance(date, datetime):
            obs_dt = date.date()
        elif isinstance(date, dt_date):
            obs_dt = date
        else:
            raise WeatherAPIError(f"Invalid date format: {date}")

        # Compute 7-day window ending on T: [T-6, T]
        start_dt = obs_dt - timedelta(days=6)
        start_date_str = start_dt.strftime("%Y-%m-%d")
        end_date_str = obs_dt.strftime("%Y-%m-%d")

        cache_key = (district, start_date_str, end_date_str)
        if cache_key in self._cache:
            return self._cache[cache_key]

        lat, lon = self._get_district_coordinates(district)

        # 1. Try Historical Archive API
        daily = self._fetch_from_endpoint(self.archive_url, lat, lon, start_date_str, end_date_str)

        # 2. Try Forecast API fallback (for recent past dates within 92 days)
        if daily is None or "precipitation_sum" not in daily or len(daily.get("precipitation_sum", [])) == 0:
            daily = self._fetch_from_endpoint(self.forecast_url, lat, lon, start_date_str, end_date_str)

        if daily is None:
            raise WeatherAPIError(
                f"Failed to retrieve weather data from Open-Meteo API for {district} "
                f"across window {start_date_str} to {end_date_str}."
            )

        # Enforce Zero Temporal Leakage: verify all returned daily dates <= obs_dt
        dates_retrieved = daily.get("time", [])
        for d_str in dates_retrieved:
            d_obj = datetime.strptime(str(d_str)[:10], "%Y-%m-%d").date()
            if d_obj > obs_dt:
                raise WeatherAPIError(
                    f"Temporal leakage violation: Open-Meteo returned future date {d_str} "
                    f"beyond observation date {end_date_str}."
                )

        precip_list = daily.get("precipitation_sum", [])
        temp_list = [t for t in daily.get("temperature_2m_mean", []) if t is not None]
        rh_list = [r for r in daily.get("relative_humidity_2m_mean", []) if r is not None]
        solar_list = [s for s in daily.get("shortwave_radiation_sum", []) if s is not None]

        # Calculate 7-day cumulative rainfall
        valid_precip = [float(p) for p in precip_list if p is not None]
        weekly_rainfall = round(sum(valid_precip), 2) if valid_precip else 0.0

        avg_temp = round(float(np.mean(temp_list)), 2) if temp_list else 26.0
        avg_rh = round(float(np.mean(rh_list)), 2) if rh_list else 80.0
        avg_solar = round(float(np.mean(solar_list)), 2) if solar_list else 18.0

        result = {
            "Date": end_date_str,
            "District": district,
            "Weekly_Rainfall_mm": weekly_rainfall,
            "Avg_Temperature_C": avg_temp,
            "Relative_Humidity_pct": avg_rh,
            "Solar_Radiation_MJ_m2_day": avg_solar,
            "Data_Source": DATA_SOURCE_API,
            "Source": "Open-Meteo",
            "Status": "SUCCESS",
            "Window_Start": start_date_str,
            "Window_End": end_date_str,
            "Daily_Precipitation": valid_precip,
            "Daily_Dates": dates_retrieved,
        }

        # Cache result
        self._cache[cache_key] = result
        return result

    def get_weather_range(
        self, district: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """Fetch range of weekly weather observations."""
        start_dt = datetime.strptime(str(start_date)[:10], "%Y-%m-%d")
        end_dt = datetime.strptime(str(end_date)[:10], "%Y-%m-%d")

        records = []
        cur_dt = start_dt
        while cur_dt <= end_dt:
            rec = self.get_weather_for_date(district, cur_dt.strftime("%Y-%m-%d"))
            records.append(rec)
            cur_dt += timedelta(days=7)

        return records

