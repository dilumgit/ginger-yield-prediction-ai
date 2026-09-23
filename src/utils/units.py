"""Unit Conversion Utilities for Farmer-Facing Displays.

This module provides unit conversion routines to translate research-level metric
yield estimates (metric tons per hectare, t/ha) to total expected harvest quantities
in kilograms (kg) for a farmer's actual cultivated land holding (in acres).

Conversion Rationale:
- 1 hectare = 2.47105 acres (canonical agronomic constant)
- 1 metric tonne = 1000 kg
- Formula:
    total_yield_kg = (predicted_t_per_ha * 1000 * land_area_acres) / 2.47105
"""

from typing import Optional, Union

# Agronomic unit conversion constants
ACRES_PER_HECTARE: float = 2.47105
KG_PER_TONNE: float = 1000.0


def total_yield_kg(
    predicted_t_per_ha: Union[float, int],
    land_area_acres: Union[float, int],
) -> float:
    """Convert yield in metric tons per hectare (t/ha) to total harvest in kg for a given acreage.

    Parameters
    ----------
    predicted_t_per_ha : float or int
        Model yield forecast in metric tons per hectare (t/ha).
    land_area_acres : float or int
        Cultivated land area in acres.

    Returns
    -------
    float
        Total expected crop harvest in kilograms (kg).

    Examples
    --------
    >>> total_yield_kg(12.93, 0.5)
    2616.2967969082
    >>> total_yield_kg(12.0, 1.0)
    4856.235199611501
    """
    if land_area_acres is None or land_area_acres <= 0:
        return 0.0
    if predicted_t_per_ha is None:
        return 0.0

    return (float(predicted_t_per_ha) * KG_PER_TONNE * float(land_area_acres)) / ACRES_PER_HECTARE


def yield_delta_kg(
    prediction_delta_t_ha: Optional[Union[float, int]],
    land_area_acres: Union[float, int],
) -> Optional[float]:
    """Convert a yield prediction shift (delta in t/ha) to total harvest shift in kg.

    Parameters
    ----------
    prediction_delta_t_ha : float or None
        Difference in predicted yield relative to previous forecast (t/ha).
    land_area_acres : float or int
        Cultivated land area in acres.

    Returns
    -------
    Optional[float]
        Harvest shift in kilograms (kg), or None if delta is not available.
    """
    if prediction_delta_t_ha is None or land_area_acres is None or land_area_acres <= 0:
        return None

    return (float(prediction_delta_t_ha) * KG_PER_TONNE * float(land_area_acres)) / ACRES_PER_HECTARE


def format_harvest_kg(kg_val: Union[float, int]) -> str:
    """Format kilogram values with comma grouping for farmer display.

    Parameters
    ----------
    kg_val : float or int
        Kilogram value.

    Returns
    -------
    str
        Formatted string, e.g. '2,615 kg'.
    """
    if kg_val is None:
        return "0 kg"
    return f"{round(kg_val):,} kg"
