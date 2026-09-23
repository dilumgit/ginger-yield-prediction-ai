"""Farmer-Friendly Irrigation Input & Conversion Engine.

This module allows farmers to record irrigation events simply using pump running time
(hours) without measuring water liters. It handles optional pump capacity to estimate
irrigation depth in mm, supports observable runtime fallback when pump capacity is unknown,
and maintains an immutable event history log.

Conversion Formulas:
- Water Volume (L) = Pump Runtime (hours) * Pump Capacity (L/hour)
- Land Area (m²) = Land Size (Acres) * 4046.86 m²/acre
- Estimated Irrigation Depth (mm) = Water Volume (L) / Land Area (m²)
  (Since 1 mm water depth = 1 L/m²)
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.utils.config import DATA_SOURCE_FARMER, SQM_PER_ACRE


def runtime_to_irrigation_depth_mm(
    runtime_hours: float,
    pump_capacity_lph: Optional[float],
    land_size_acres: float,
) -> Optional[float]:
    """Convert pump runtime (hours) to estimated irrigation depth (mm).

    Parameters
    ----------
    runtime_hours : float
        Total duration pump was operated in hours.
    pump_capacity_lph : float or None
        Water delivery rate in Liters per hour (L/h).
        If None, conversion cannot occur and returns None (fallback to runtime).
    land_size_acres : float
        Land size under cultivation in acres.

    Returns
    -------
    Optional[float]
        Estimated irrigation depth in mm, or None if pump capacity is unknown.
    """
    if pump_capacity_lph is None or pump_capacity_lph <= 0:
        return None

    if land_size_acres <= 0 or runtime_hours <= 0:
        return 0.0

    water_volume_liters = float(runtime_hours) * float(pump_capacity_lph)
    area_sqm = float(land_size_acres) * SQM_PER_ACRE

    # 1 L per 1 m² = 1 mm depth
    depth_mm = water_volume_liters / area_sqm
    return round(depth_mm, 4)


@dataclass
class IrrigationEvent:
    """Dataclass representing a single farmer-reported irrigation event."""

    date: str  # Format: "YYYY-MM-DD"
    runtime_hours: float
    water_source: str = "Well"
    irrigation_method: str = "Sprinkler"
    notes: Optional[str] = None
    data_source: str = DATA_SOURCE_FARMER

    def __post_init__(self):
        """Validate input parameters upon creation."""
        if self.runtime_hours < 0:
            raise ValueError(f"Pump runtime cannot be negative: {self.runtime_hours} hours.")
        if self.runtime_hours > 24:
            raise ValueError(f"Pump runtime exceeds 24 hours in a single day: {self.runtime_hours} hours.")
        # Validate date string format
        try:
            datetime.strptime(str(self.date)[:10], "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format for irrigation event: '{self.date}'. Expected YYYY-MM-DD.") from e


class IrrigationEventLog:
    """Immutable event history logger for farm irrigation management."""

    def __init__(self, events: Optional[List[IrrigationEvent]] = None):
        """Initialize event log."""
        self._events: List[IrrigationEvent] = []
        if events:
            for ev in events:
                self.add_event(ev)

    @property
    def events(self) -> List[IrrigationEvent]:
        """Return list of recorded irrigation events."""
        return self._events[:]

    def add_event(
        self,
        event: Union[IrrigationEvent, Dict[str, Any]],
    ) -> None:
        """Record and store a new irrigation event in the history log.

        Parameters
        ----------
        event : IrrigationEvent or Dict
            Irrigation event object or dictionary with event fields.
        """
        if isinstance(event, dict):
            ev_obj = IrrigationEvent(**event)
        elif isinstance(event, IrrigationEvent):
            ev_obj = event
        else:
            raise TypeError(f"Expected IrrigationEvent or dict, got {type(event)}")

        self._events.append(ev_obj)
        # Keep events sorted chronologically
        self._events.sort(key=lambda x: str(x.date)[:10])

    def get_events(
        self,
        as_of_date: Optional[str] = None,
        window_days: Optional[int] = None,
    ) -> List[IrrigationEvent]:
        """Retrieve events filtered up to `as_of_date` (preventing future data leakage).

        Parameters
        ----------
        as_of_date : str, optional
            Cutoff date string (YYYY-MM-DD). Only events on or before this date are returned.
        window_days : int, optional
            If specified, only returns events within [as_of_date - window_days, as_of_date].

        Returns
        -------
        List[IrrigationEvent]
            Filtered list of irrigation events.
        """
        if not self._events:
            return []

        filtered = self._events[:]
        if as_of_date is not None:
            cutoff_dt = datetime.strptime(str(as_of_date)[:10], "%Y-%m-%d")
            filtered = [
                ev for ev in filtered
                if datetime.strptime(str(ev.date)[:10], "%Y-%m-%d") <= cutoff_dt
            ]

            if window_days is not None:
                start_dt = cutoff_dt - timedelta(days=window_days)
                filtered = [
                    ev for ev in filtered
                    if datetime.strptime(str(ev.date)[:10], "%Y-%m-%d") >= start_dt
                ]

        return filtered

    def get_recent_runtime_hours(
        self, as_of_date: str, window_days: int = 7
    ) -> float:
        """Calculate total pump runtime hours within a recent rolling window."""
        events = self.get_events(as_of_date=as_of_date, window_days=window_days)
        return float(sum(ev.runtime_hours for ev in events))

    def get_cumulative_runtime_hours(self, as_of_date: str) -> float:
        """Calculate all-time cumulative pump runtime hours up to `as_of_date`."""
        events = self.get_events(as_of_date=as_of_date)
        return float(sum(ev.runtime_hours for ev in events))

    def get_estimated_recent_depth_mm(
        self,
        as_of_date: str,
        land_size_acres: float,
        pump_capacity_lph: Optional[float],
        window_days: int = 7,
    ) -> Optional[float]:
        """Estimate recent irrigation depth in mm over a rolling window."""
        recent_runtime = self.get_recent_runtime_hours(as_of_date, window_days=window_days)
        return runtime_to_irrigation_depth_mm(recent_runtime, pump_capacity_lph, land_size_acres)

    def get_estimated_cumulative_depth_mm(
        self,
        as_of_date: str,
        land_size_acres: float,
        pump_capacity_lph: Optional[float],
    ) -> Optional[float]:
        """Estimate total cumulative irrigation depth in mm up to `as_of_date`."""
        cum_runtime = self.get_cumulative_runtime_hours(as_of_date)
        return runtime_to_irrigation_depth_mm(cum_runtime, pump_capacity_lph, land_size_acres)

    def to_dataframe(self) -> pd.DataFrame:
        """Export full irrigation event history as a pandas DataFrame."""
        if not self._events:
            return pd.DataFrame(columns=["date", "runtime_hours", "water_source", "irrigation_method", "notes", "data_source"])
        return pd.DataFrame([asdict(ev) for ev in self._events])

    def __len__(self) -> int:
        return len(self._events)
