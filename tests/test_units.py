"""Unit Tests for Centralized Unit Conversion Module (src/utils/units.py).

Verifies:
1. 1 t/ha on 1 acre conversion.
2. 12 t/ha on 0.5 acres conversion.
3. 12.93 t/ha on 0.5 acres conversion (~2,615 - 2,616 kg).
4. 16.63 t/ha on 1 acre conversion.
5. Zero-acre validation and edge cases (None, negative).
6. Decimal acre values (0.25, 0.50, 0.75, 1.25, 2.50, 5.00 acres).
7. Prediction delta conversion (yield_delta_kg).
8. Formatting helper (format_harvest_kg).
"""

import pytest
from src.utils.units import (
    ACRES_PER_HECTARE,
    KG_PER_TONNE,
    format_harvest_kg,
    total_yield_kg,
    yield_delta_kg,
)


def test_conversion_constants():
    """Verify conversion constants are correctly set."""
    assert ACRES_PER_HECTARE == 2.47105
    assert KG_PER_TONNE == 1000.0


def test_1_t_per_ha_on_1_acre():
    """Verify 1 t/ha on 1 acre = 1 * 1000 * 1 / 2.47105 = 404.68626... kg."""
    expected = (1.0 * 1000.0 * 1.0) / 2.47105
    result = total_yield_kg(predicted_t_per_ha=1.0, land_area_acres=1.0)
    assert result == pytest.approx(expected, rel=1e-5)
    assert round(result) == 405


def test_12_t_per_ha_on_0_point_5_acres():
    """Verify 12 t/ha on 0.5 acres = 12 * 1000 * 0.5 / 2.47105 = 2428.117... kg."""
    expected = (12.0 * 1000.0 * 0.5) / 2.47105
    result = total_yield_kg(predicted_t_per_ha=12.0, land_area_acres=0.5)
    assert result == pytest.approx(expected, rel=1e-5)
    assert round(result) == 2428


def test_12_point_93_t_per_ha_on_0_point_5_acres():
    """Verify 12.93 t/ha on 0.5 acres = 12.93 * 1000 * 0.5 / 2.47105 = 2616.296... kg (~2615-2616 kg)."""
    expected = (12.93 * 1000.0 * 0.5) / 2.47105
    result = total_yield_kg(predicted_t_per_ha=12.93, land_area_acres=0.5)
    assert result == pytest.approx(expected, rel=1e-5)
    assert round(result) == 2616
    assert abs(result - 2615.0) < 2.0  # approximately 2615 kg as specified


def test_16_point_63_t_per_ha_on_1_acre():
    """Verify 16.63 t/ha on 1 acre = 16.63 * 1000 * 1 / 2.47105 = 6729.932... kg."""
    expected = (16.63 * 1000.0 * 1.0) / 2.47105
    result = total_yield_kg(predicted_t_per_ha=16.63, land_area_acres=1.0)
    assert result == pytest.approx(expected, rel=1e-5)
    assert round(result) == 6730


def test_zero_and_invalid_acre_validation():
    """Verify 0 acres, negative acres, None, and negative predictions are safely handled."""
    assert total_yield_kg(predicted_t_per_ha=12.0, land_area_acres=0.0) == 0.0
    assert total_yield_kg(predicted_t_per_ha=12.0, land_area_acres=-1.0) == 0.0
    assert total_yield_kg(predicted_t_per_ha=12.0, land_area_acres=None) == 0.0
    assert total_yield_kg(predicted_t_per_ha=None, land_area_acres=1.0) == 0.0


def test_decimal_acre_values():
    """Verify various decimal land sizes scale linearly with harvest weight."""
    yield_t_ha = 12.0

    # 0.25 acres
    kg_0_25 = total_yield_kg(yield_t_ha, 0.25)
    assert kg_0_25 == pytest.approx((12.0 * 1000.0 * 0.25) / 2.47105, rel=1e-5)
    assert round(kg_0_25) == 1214

    # 0.50 acres
    kg_0_50 = total_yield_kg(yield_t_ha, 0.50)
    assert kg_0_50 == pytest.approx((12.0 * 1000.0 * 0.50) / 2.47105, rel=1e-5)
    assert round(kg_0_50) == 2428

    # 0.75 acres
    kg_0_75 = total_yield_kg(yield_t_ha, 0.75)
    assert kg_0_75 == pytest.approx((12.0 * 1000.0 * 0.75) / 2.47105, rel=1e-5)
    assert round(kg_0_75) == 3642

    # 1.00 acre
    kg_1_00 = total_yield_kg(yield_t_ha, 1.00)
    assert kg_1_00 == pytest.approx((12.0 * 1000.0 * 1.00) / 2.47105, rel=1e-5)
    assert round(kg_1_00) == 4856

    # 2.00 acres
    kg_2_00 = total_yield_kg(yield_t_ha, 2.00)
    assert kg_2_00 == pytest.approx((12.0 * 1000.0 * 2.00) / 2.47105, rel=1e-5)
    assert round(kg_2_00) == 9712

    # Verify exact doubling
    assert kg_0_50 == pytest.approx(kg_0_25 * 2.0, rel=1e-5)
    assert kg_1_00 == pytest.approx(kg_0_50 * 2.0, rel=1e-5)
    assert kg_2_00 == pytest.approx(kg_1_00 * 2.0, rel=1e-5)


def test_yield_delta_kg_calculation():
    """Verify prediction delta conversion to kg harvest shift."""
    # From 16.63 to 12.93 on 0.5 acres -> delta = -3.70 t/ha -> -749 kg
    delta_t_ha = 12.93 - 16.63  # -3.70
    delta_kg = yield_delta_kg(delta_t_ha, 0.5)
    assert delta_kg == pytest.approx((-3.70 * 1000 * 0.5) / 2.47105, rel=1e-5)
    assert round(delta_kg) == -749

    # Positive shift: +0.49 t/ha on 1.0 acre -> +198 kg
    pos_delta_kg = yield_delta_kg(0.49, 1.0)
    assert round(pos_delta_kg) == 198

    # None handling
    assert yield_delta_kg(None, 1.0) is None
    assert yield_delta_kg(0.49, 0.0) is None


def test_format_harvest_kg():
    """Verify formatting string helper."""
    assert format_harvest_kg(2616.29) == "2,616 kg"
    assert format_harvest_kg(4856.23) == "4,856 kg"
    assert format_harvest_kg(0) == "0 kg"
    assert format_harvest_kg(None) == "0 kg"
