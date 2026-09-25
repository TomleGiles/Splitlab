"""Tests de la fiche de course et de l'endpoint de démonstration."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.sheet import metric_out, splits_out, stroke_reading
from cv.domain import NOT_MEASURABLE, MetricValue
from cv.metrics import SectionMetrics


def section(stroke_rate: float, stroke_length: float) -> SectionMetrics:
    return SectionMetrics(
        stroke_rate=MetricValue(stroke_rate, 1.0),
        stroke_length=MetricValue(stroke_length, 1.0),
        stroke_index=MetricValue(stroke_length * 2.0, 1.0),
    )


@pytest.mark.parametrize(
    ("inbound_sr", "inbound_sl", "expected"),
    [
        (49.0, 1.80, "signe de fatigue"),  # SR −2 %, SL −10 %
        (45.0, 1.98, "baisse de rythme"),  # SR −10 %, SL −1 %
        (55.0, 1.80, "compenser"),  # SR +10 %, SL −10 %
        (50.5, 2.02, "stables"),  # SR +1 %, SL +1 %
    ],
)
def test_stroke_reading_diagnosis(inbound_sr: float, inbound_sl: float, expected: str) -> None:
    reading = stroke_reading(section(50.0, 2.0), section(inbound_sr, inbound_sl))

    assert reading.diagnosis is not None
    assert expected in reading.diagnosis


def test_stroke_reading_without_inbound_section_has_no_diagnosis() -> None:
    missing = SectionMetrics(NOT_MEASURABLE, NOT_MEASURABLE, NOT_MEASURABLE)

    reading = stroke_reading(section(50.0, 2.0), missing)

    assert reading.diagnosis is None
    assert reading.stroke_rate_change_pct is None


def test_low_confidence_is_flagged_to_check_but_missing_value_is_not() -> None:
    assert metric_out(MetricValue(1.0, 0.69)).to_check
    assert not metric_out(MetricValue(1.0, 0.7)).to_check
    assert not metric_out(NOT_MEASURABLE).to_check


def test_split_segments() -> None:
    splits = {
        15.0: MetricValue(6.0, 1.0),
        25.0: MetricValue(11.0, 1.0),
        35.0: NOT_MEASURABLE,
        50.0: MetricValue(23.0, 1.0),
    }

    out = splits_out(splits)

    assert [s.segment_time for s in out] == [6.0, 5.0, None, None]
    assert out[0].segment_speed == pytest.approx(2.5)
    assert out[1].segment_speed == pytest.approx(2.0)


def test_demo_race_endpoint_returns_a_plausible_sheet() -> None:
    response = TestClient(app).get("/api/demo/race")

    assert response.status_code == 200
    sheet = response.json()
    assert sheet["is_demo"] is True
    assert [s["distance"] for s in sheet["splits"]] == [15.0, 25.0, 35.0, 50.0]
    assert 20.0 < sheet["splits"][-1]["time"]["value"] < 30.0
    assert len(sheet["velocity_profile"]) == 101
    assert "fatigue" in sheet["stroke_reading"]["diagnosis"]
