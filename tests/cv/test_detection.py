"""Tests de cv/detection.py : masques élémentaires et vidéo synthétique vue de biais."""

import numpy as np
import pytest

from cv.detection import (
    PIXELS_PER_M,
    X_MARGIN_M,
    LaneDetections,
    LaneStrip,
    detect_frames,
    front_position,
    locate,
)
from tests.cv.synthetic_video import FPS, render, synthetic_calibration

TOLERANCE_M = 0.1  # deux colonnes de bande : flou de la projection

# --- locate : masques de bande dessinés à la main ----------------------------------


def band(*blobs: tuple[float, float]) -> np.ndarray:
    """Masque de bande (30 × 520 px) avec des zones pleines entre x_start et x_end (m)."""
    mask = np.zeros((30, 520), dtype=bool)
    for x_start, x_end in blobs:
        a = round((x_start + X_MARGIN_M) * PIXELS_PER_M)
        b = round((x_end + X_MARGIN_M) * PIXELS_PER_M)
        mask[10:20, a:b] = True
    return mask


def test_empty_lane_gives_no_observation() -> None:
    assert locate(band()) is None


def test_largest_zone_wins_and_confidence_is_its_share() -> None:
    observation = locate(band((10.0, 12.0), (20.0, 20.5)))

    assert observation is not None
    assert observation.x_min == pytest.approx(10.0)
    assert observation.x_max == pytest.approx(12.0)
    assert observation.x_center == pytest.approx(11.0, abs=0.05)
    assert observation.confidence == pytest.approx(2.0 / 2.5)


def test_close_zones_are_merged_into_one_swimmer() -> None:
    observation = locate(band((10.0, 11.0), (11.3, 12.0)))  # bras détaché du corps

    assert observation is not None
    assert (observation.x_min, observation.x_max) == pytest.approx((10.0, 12.0))
    assert observation.confidence == pytest.approx(1.0)


def test_tiny_zone_is_ignored() -> None:
    assert locate(band((10.0, 10.2))) is None


# --- Vidéo synthétique ---------------------------------------------------------------

START = 0.4  # s : avant, le couloir est vide


def our_front(t: float) -> float | None:
    return None if t < START else 1.5 + 2.0 * (t - START)


def neighbour_front(t: float) -> float | None:
    return None if t < START else 1.5 + 2.4 * (t - START)  # plus rapide, couloir 5


def test_swimmer_is_located_in_oblique_view_and_neighbour_ignored() -> None:
    calibration = synthetic_calibration()
    frames = render([(4, our_front), (5, neighbour_front)], duration=6.0)

    detections = detect_frames(frames, LaneStrip.for_lane(calibration, lane=4))

    t = np.arange(detections.x_max.size) / FPS
    empty = t < START
    assert np.all(np.isnan(detections.x_max[empty]))
    swimming = t >= START
    expected = np.array([our_front(ti) for ti in t[swimming]], dtype=np.float64)
    error = detections.x_max[swimming] - expected
    assert np.max(np.abs(error)) < TOLERANCE_M
    assert np.all(detections.confidence[swimming] > 0.9)


# --- Avant du nageur de part et d'autre du demi-tour ----------------------------------


def test_front_is_leading_edge_in_each_direction() -> None:
    center = 23.0 - np.abs(np.arange(-10, 11)) * 1.0  # demi-tour à l'image 10
    detections = LaneDetections(
        x_min=center - 1.0,
        x_max=center + 1.0,
        x_center=center,
        confidence=np.ones_like(center),
    )

    front = front_position(detections)

    assert front[:11] == pytest.approx(center[:11] + 1.0)
    assert front[11:] == pytest.approx(center[11:] - 1.0)


def test_front_is_nan_where_swimmer_is_not_seen() -> None:
    nan = np.full(3, np.nan)
    detections = LaneDetections(nan, nan, nan, np.zeros(3))

    assert np.all(np.isnan(front_position(detections)))
