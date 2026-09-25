"""Tests de cv/trajectory.py sur positions brutes synthétiques à résultat connu."""

import numpy as np
import pytest
from numpy.typing import NDArray

from cv.domain import Event, EventType, FloatArray
from cv.metrics import compute_race_metrics
from cv.trajectory import (
    EDGE_CONFIDENCE_FACTOR,
    INTERPOLATED_CONFIDENCE_FACTOR,
    build_trajectory,
)

FPS = 50.0
POOL = 25.0


def frames(t_end: float, t_start: float = 0.0) -> FloatArray:
    return np.linspace(t_start, t_end, round((t_end - t_start) * FPS) + 1)


def pool_x(d: FloatArray, apex: float = POOL) -> FloatArray:
    """x du point suivi pour une distance « idéale » d, avec demi-tour à x = apex."""
    return np.where(d <= apex, d, 2 * apex - d)


def full_confidence(t: FloatArray) -> FloatArray:
    return np.ones_like(t)


def at(t: FloatArray, instant: float) -> int:
    return int(np.argmin(np.abs(t - instant)))


HALF_WINDOW = 25  # échantillons : fenêtre de lissage de 1 s à 50 fps


def length_edges(size: int, last_outbound: int) -> NDArray[np.bool_]:
    """Demi-fenêtre au début et à la fin de chaque longueur."""
    index = np.arange(size)
    near_turn = np.abs(index - (last_outbound + 0.5)) < HALF_WINDOW
    return (index < HALF_WINDOW) | near_turn | (index >= size - HALF_WINDOW)


# --- Vitesse constante 2 m/s, sans bruit ---------------------------------------------


def test_linear_motion_is_reconstructed_exactly() -> None:
    t = frames(25.0)
    traj = build_trajectory(t, pool_x(2.0 * t), full_confidence(t))

    assert traj.d == pytest.approx(2.0 * t, abs=1e-9)
    assert traj.v == pytest.approx(np.full_like(t, 2.0), abs=1e-9)
    edges = length_edges(t.size, last_outbound=at(t, 12.5))
    assert np.all(traj.confidence[edges] == EDGE_CONFIDENCE_FACTOR)
    assert np.all(traj.confidence[~edges] == 1.0)


def test_turn_before_wall_makes_d_jump_but_keeps_speed_positive() -> None:
    # Le point suivi fait demi-tour à x = 23,6 m (t = 11,8 s) au lieu du mur.
    t = frames(25.0)
    apex_t, apex_x = 11.8, 23.6
    x = np.where(t <= apex_t, 2.0 * t, apex_x - 2.0 * (t - apex_t))

    traj = build_trajectory(t, x, full_confidence(t))

    outbound = t <= apex_t
    assert traj.d[outbound] == pytest.approx(2.0 * t[outbound], abs=1e-9)
    assert traj.d[~outbound] == pytest.approx(
        2 * POOL - apex_x + 2.0 * (t[~outbound] - apex_t), abs=1e-9
    )
    assert traj.v == pytest.approx(np.full_like(t, 2.0), abs=1e-9)


def test_explicit_turn_time_overrides_estimate() -> None:
    # Demi-tour estimé à 12,5 s (x maximal) ; l'utilisateur l'a placé à 12,0 s.
    t = frames(25.0)
    traj = build_trajectory(t, pool_x(2.0 * t), full_confidence(t), turn_t=12.0)

    assert traj.d[t <= 12.0] == pytest.approx(traj.x[t <= 12.0])
    assert traj.d[t > 12.0] == pytest.approx(2 * POOL - traj.x[t > 12.0])


# --- Détections manquantes et aberrantes ---------------------------------------------


def test_short_gap_is_interpolated_with_reduced_confidence() -> None:
    t = frames(25.0)
    x = pool_x(2.0 * t)
    gap = (t > 5.0) & (t < 5.3)
    x[gap] = np.nan

    traj = build_trajectory(t, x, full_confidence(t))

    assert traj.d == pytest.approx(2.0 * t, abs=1e-9)
    assert traj.confidence[gap] == pytest.approx(INTERPOLATED_CONFIDENCE_FACTOR)
    edges = length_edges(t.size, last_outbound=at(t, 12.5))
    assert np.all(traj.confidence[~gap & ~edges] == 1.0)


def test_long_gap_is_interpolated_with_zero_confidence() -> None:
    t = frames(25.0)
    x = pool_x(2.0 * t)
    gap = (t > 8.0) & (t < 9.5)
    x[gap] = np.nan

    traj = build_trajectory(t, x, full_confidence(t))

    assert not np.any(np.isnan(traj.d))
    assert np.all(traj.confidence[gap] == 0.0)


def test_outlier_detection_is_rejected() -> None:
    t = frames(25.0)
    x = pool_x(2.0 * t)
    spike = at(t, 6.0)
    x[spike] += 3.0  # saut vers le nageur de la ligne voisine

    traj = build_trajectory(t, x, full_confidence(t))

    assert traj.d[spike] == pytest.approx(12.0, abs=1e-9)
    assert traj.confidence[spike] == pytest.approx(INTERPOLATED_CONFIDENCE_FACTOR)


def test_unobserved_start_and_end_are_trimmed_not_extrapolated() -> None:
    t = frames(25.0)
    x = pool_x(2.0 * t)
    x[t < 0.5] = np.nan
    x[t > 24.0] = np.nan

    traj = build_trajectory(t, x, full_confidence(t))

    assert traj.t[0] == pytest.approx(0.5)
    assert traj.t[-1] == pytest.approx(24.0)


def test_zero_confidence_detections_are_ignored() -> None:
    t = frames(25.0)
    x = pool_x(2.0 * t)
    confidence = full_confidence(t)
    x[at(t, 3.0)] = 40.0
    confidence[at(t, 3.0)] = 0.0

    traj = build_trajectory(t, x, confidence)

    assert traj.d[at(t, 3.0)] == pytest.approx(6.0, abs=1e-9)


def test_irregular_frame_times_are_rejected() -> None:
    t = np.array([0.0, 0.02, 0.05, 0.06])
    with pytest.raises(ValueError, match="pas constant"):
        build_trajectory(t, t, full_confidence(t))


def test_too_few_detections_are_rejected() -> None:
    t = frames(1.0)
    x = np.full_like(t, np.nan)
    x[0] = 0.0
    with pytest.raises(ValueError, match="détections"):
        build_trajectory(t, x, full_confidence(t))


# --- Nageur qui décélère, positions bruitées : cibles MVP de CLAUDE.md ----------------

V0, DECEL, NOISE_M = 2.4, 0.02, 0.05


def time_at(distance: float) -> float:
    return float((V0 - np.sqrt(V0**2 - 2 * DECEL * distance)) / DECEL)


def noisy_race() -> tuple[FloatArray, FloatArray]:
    rng = np.random.default_rng(seed=7)
    t = frames(24.0)
    d = V0 * t - DECEL * t**2 / 2
    return t, pool_x(d) + rng.normal(0.0, NOISE_M, t.size)


def test_noisy_positions_give_splits_within_mvp_targets() -> None:
    t, x = noisy_race()
    traj = build_trajectory(t, x, full_confidence(t))
    events = [
        Event(EventType.WALL_IN, time_at(25.0), frame=0, confidence=1.0),
        Event(EventType.FINISH, time_at(50.0), frame=0, confidence=1.0),
    ]

    metrics = compute_race_metrics(traj, events)

    # Cibles MVP : passages ±0,05 s, temps de virage ±0,1 s.
    assert metrics.splits[15.0].value == pytest.approx(time_at(15.0), abs=0.05)
    assert metrics.splits[35.0].value == pytest.approx(time_at(35.0), abs=0.05)
    assert metrics.turn_time.value == pytest.approx(time_at(30.0) - time_at(20.0), abs=0.1)


def test_noisy_positions_give_accurate_speed_where_confident() -> None:
    t, x = noisy_race()
    traj = build_trajectory(t, x, full_confidence(t))

    error = traj.v - (V0 - DECEL * t)
    confident = traj.confidence == 1.0
    assert np.sqrt(np.mean(error[confident] ** 2)) < 0.05
    assert np.max(np.abs(error[confident])) < 0.1
    # Aux bords des longueurs, l'erreur est plus grande et la confiance abaissée.
    assert np.all(traj.confidence[~confident] == EDGE_CONFIDENCE_FACTOR)
