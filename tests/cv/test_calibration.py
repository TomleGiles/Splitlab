"""Tests de cv/calibration.py."""

from pathlib import Path

import numpy as np
import pytest

from cv.calibration import CalibrationError, calibrate, load_calibration, save_calibration
from cv.calibration import transform as apply_homography
from tests.cv.synthetic_video import calibration_points


def test_homography_maps_clicked_points_to_pool_coordinates() -> None:
    image, pool = calibration_points()

    calibration = calibrate(image, pool)

    assert apply_homography(calibration.homography, image) == pytest.approx(pool, abs=1e-6)
    assert calibration.reprojection_error < 1e-6


def test_misplaced_click_is_rejected_and_named() -> None:
    image, pool = calibration_points()
    pool[4] += [1.0, 0.0]  # repère 5 saisi avec 1 m d'erreur

    with pytest.raises(CalibrationError, match="repère 5"):
        calibrate(image, pool)


def test_at_least_four_points_are_required() -> None:
    image, pool = calibration_points()
    with pytest.raises(CalibrationError, match="4 repères"):
        calibrate(image[:3], pool[:3])


def test_lane_band_follows_lane_width_and_first_lane() -> None:
    image, pool = calibration_points()
    calibration = calibrate(image, pool, lane_width=2.0, first_lane=0)

    assert calibration.lane_band(0) == (0.0, 2.0)
    assert calibration.lane_band(4) == (8.0, 10.0)
    with pytest.raises(ValueError, match="premier couloir"):
        calibration.lane_band(-1)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    image, pool = calibration_points()
    calibration = calibrate(image, pool, lane_width=2.5, first_lane=1)
    path = tmp_path / "calib.json"

    save_calibration(calibration, path)
    loaded = load_calibration(path)

    assert np.allclose(loaded.homography, calibration.homography)
    assert loaded.lane_band(4) == calibration.lane_band(4)
