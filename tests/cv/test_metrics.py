"""Tests de cv/metrics.py sur trajectoires synthétiques à résultat connu analytiquement."""

from collections.abc import Callable
from dataclasses import fields

import numpy as np
import pytest

from cv.domain import NOT_MEASURABLE, Event, EventType, FloatArray, Trajectory
from cv.metrics import (
    PROFILE_STEP,
    RaceMetrics,
    compute_race_metrics,
    velocity_profile,
)

FPS = 50.0
POOL = 25.0


def make_trajectory(
    distance: Callable[[FloatArray], FloatArray],
    speed: Callable[[FloatArray], FloatArray],
    t_end: float,
    confidence: Callable[[FloatArray], FloatArray] | None = None,
) -> Trajectory:
    t = np.linspace(0.0, t_end, round(t_end * FPS) + 1)
    d = distance(t)
    x = np.where(d <= POOL, d, 2 * POOL - d)
    c = np.ones_like(t) if confidence is None else confidence(d)
    return Trajectory(t=t, x=x, d=d, v=speed(t), confidence=c)


def event(event_type: EventType, t: float, confidence: float = 1.0, manual: bool = False) -> Event:
    return Event(
        type=event_type,
        t=t,
        frame=round(t * FPS),
        confidence=confidence,
        manually_corrected=manual,
    )


def cycles(t_first: float, period: float, count: int) -> list[Event]:
    return [event(EventType.STROKE_CYCLE, t_first + k * period) for k in range(count)]


# --- Nageur à vitesse constante : 2 m/s, virage à 12,5 s, arrivée à 25 s -------------

SPEED = 2.0


def constant_speed_trajectory(t_end: float = 25.0) -> Trajectory:
    return make_trajectory(lambda t: SPEED * t, lambda t: np.full_like(t, SPEED), t_end)


def constant_speed_events() -> list[Event]:
    return [
        event(EventType.START_SIGNAL, 0.0),
        event(EventType.ENTRY, 1.0),
        event(EventType.BREAKOUT, 3.0),  # x = 6 m
        *cycles(3.0, 1.2, 8),  # 7 cycles, SR = 50 cycles/min
        event(EventType.WALL_IN, 12.5),
        event(EventType.WALL_OUT, 13.0),
        event(EventType.BREAKOUT, 16.0),  # d = 32 m, x = 18 m → 7 m depuis le mur
        *cycles(16.0, 1.25, 7),  # 6 cycles, SR = 48 cycles/min
        event(EventType.FINISH, 25.0),
    ]


@pytest.fixture
def constant_metrics() -> RaceMetrics:
    return compute_race_metrics(constant_speed_trajectory(), constant_speed_events())


def test_constant_speed_splits(constant_metrics: RaceMetrics) -> None:
    values = {d: m.value for d, m in constant_metrics.splits.items()}
    assert values == pytest.approx({15.0: 7.5, 25.0: 12.5, 35.0: 17.5, 50.0: 25.0})


def test_constant_speed_underwater_distances(constant_metrics: RaceMetrics) -> None:
    assert constant_metrics.underwater_start.value == pytest.approx(6.0)
    assert constant_metrics.underwater_turn.value == pytest.approx(7.0)


def test_constant_speed_stroke_sections(constant_metrics: RaceMetrics) -> None:
    outbound, inbound = constant_metrics.sections
    assert outbound.stroke_rate.value == pytest.approx(50.0)
    assert outbound.stroke_length.value == pytest.approx(2.4)  # 2 / (50 / 60)
    assert outbound.stroke_index.value == pytest.approx(4.8)  # 2 × 2,4
    assert inbound.stroke_rate.value == pytest.approx(48.0)
    assert inbound.stroke_length.value == pytest.approx(2.5)
    assert inbound.stroke_index.value == pytest.approx(5.0)


def test_constant_speed_turn_and_finish(constant_metrics: RaceMetrics) -> None:
    assert constant_metrics.turn_time.value == pytest.approx(5.0)
    assert constant_metrics.finish_speed.value == pytest.approx(SPEED)


def test_constant_speed_velocity_profile(constant_metrics: RaceMetrics) -> None:
    profile = constant_metrics.velocity_profile
    assert profile.d.size == round(50.0 / PROFILE_STEP) + 1
    assert profile.d[[0, -1]] == pytest.approx([0.0, 50.0])
    assert profile.v == pytest.approx(np.full(profile.d.size, SPEED))


def test_reaction_time_is_never_estimated_without_block_off(
    constant_metrics: RaceMetrics,
) -> None:
    assert constant_metrics.reaction_time == NOT_MEASURABLE


def test_reaction_time_runs_from_start_signal_to_block_off() -> None:
    events = [
        event(EventType.START_SIGNAL, 0.0),
        event(EventType.BLOCK_OFF, 0.68, confidence=0.9),
    ]

    reaction = compute_race_metrics(constant_speed_trajectory(), events).reaction_time

    assert reaction.value == pytest.approx(0.68)
    assert reaction.confidence == pytest.approx(0.9)


def test_full_confidence_and_no_manual_correction(constant_metrics: RaceMetrics) -> None:
    assert constant_metrics.turn_time.confidence == 1.0
    assert not constant_metrics.sections[0].stroke_rate.manually_corrected


# --- Nageur qui décélère uniformément : d(t) = v0·t − a·t²/2 ------------------------

V0, DECEL = 2.4, 0.02


def time_at(distance: float) -> float:
    return float((V0 - np.sqrt(V0**2 - 2 * DECEL * distance)) / DECEL)


def speed_at(distance: float) -> float:
    return float(np.sqrt(V0**2 - 2 * DECEL * distance))


def test_decelerating_swimmer_matches_analytic_solution() -> None:
    traj = make_trajectory(
        lambda t: V0 * t - DECEL * t**2 / 2, lambda t: V0 - DECEL * t, t_end=24.0
    )
    events = [event(EventType.WALL_IN, time_at(25.0)), event(EventType.FINISH, time_at(50.0))]

    metrics = compute_race_metrics(traj, events)

    assert metrics.splits[15.0].value == pytest.approx(time_at(15.0), abs=1e-4)
    assert metrics.splits[35.0].value == pytest.approx(time_at(35.0), abs=1e-4)
    assert metrics.turn_time.value == pytest.approx(time_at(30.0) - time_at(20.0), abs=1e-4)
    assert metrics.finish_speed.value == pytest.approx(
        5.0 / (time_at(50.0) - time_at(45.0)), abs=1e-4
    )
    expected_profile = [speed_at(d) for d in metrics.velocity_profile.d]
    assert metrics.velocity_profile.v == pytest.approx(expected_profile, abs=1e-4)


# --- Robustesse -----------------------------------------------------------------------


def test_split_uses_first_crossing_when_distance_is_noisy() -> None:
    traj = constant_speed_trajectory()
    d = traj.d.copy()
    d[round(7.6 * FPS)] = 14.0  # retour en arrière ponctuel après le passage à 15 m
    noisy = Trajectory(t=traj.t, x=traj.x, d=d, v=traj.v, confidence=traj.confidence)

    assert compute_race_metrics(noisy, []).splits[15.0].value == pytest.approx(7.5)


def test_confidence_is_minimum_of_samples_and_events_used() -> None:
    traj = make_trajectory(
        lambda t: SPEED * t,
        lambda t: np.full_like(t, SPEED),
        t_end=25.0,
        confidence=lambda d: np.where((d > 14.0) & (d < 16.0), 0.4, 1.0),
    )
    events = constant_speed_events()
    events[3] = event(EventType.STROKE_CYCLE, events[3].t, confidence=0.7, manual=True)

    metrics = compute_race_metrics(traj, events)

    assert metrics.splits[15.0].confidence == pytest.approx(0.4)
    assert metrics.splits[35.0].confidence == pytest.approx(1.0)
    outbound, inbound = metrics.sections
    assert outbound.stroke_rate.confidence == pytest.approx(0.7)
    assert outbound.stroke_rate.manually_corrected
    assert not inbound.stroke_rate.manually_corrected


def test_missing_events_are_not_measurable_never_estimated() -> None:
    metrics = compute_race_metrics(constant_speed_trajectory(), [])

    assert metrics.splits[15.0].value == pytest.approx(7.5)
    section_values = [
        getattr(section, f.name) for section in metrics.sections for f in fields(section)
    ]
    for missing in (
        metrics.splits[25.0],
        metrics.splits[50.0],
        metrics.underwater_start,
        metrics.underwater_turn,
        metrics.finish_speed,
        *section_values,
    ):
        assert missing == NOT_MEASURABLE


def test_lost_tracking_leaves_unreached_distances_unmeasured() -> None:
    traj = constant_speed_trajectory(t_end=15.0)  # suivi perdu à d = 30 m
    events = [event(EventType.FINISH, 25.0)]

    metrics = compute_race_metrics(traj, events)

    assert metrics.splits[35.0] == NOT_MEASURABLE
    assert metrics.finish_speed == NOT_MEASURABLE
    assert metrics.turn_time.value == pytest.approx(5.0)
    profile = velocity_profile(traj)
    unreached = profile.d > 30.0
    assert np.all(np.isnan(profile.v[unreached]))
    assert np.all(profile.confidence[unreached] == 0.0)
    assert not np.any(np.isnan(profile.v[~unreached]))


def test_duplicate_unique_event_is_rejected() -> None:
    events = [event(EventType.FINISH, 24.9), event(EventType.FINISH, 25.0)]
    with pytest.raises(ValueError, match="finish"):
        compute_race_metrics(constant_speed_trajectory(), events)


def test_breakout_during_turn_is_rejected() -> None:
    events = [
        event(EventType.WALL_IN, 12.5),
        event(EventType.BREAKOUT, 12.8),
        event(EventType.WALL_OUT, 13.0),
    ]
    with pytest.raises(ValueError, match="virage"):
        compute_race_metrics(constant_speed_trajectory(), events)


def test_trajectory_requires_strictly_increasing_time() -> None:
    t = np.array([0.0, 1.0, 1.0])
    ones = np.ones(3)
    with pytest.raises(ValueError, match="croissants"):
        Trajectory(t=t, x=ones, d=ones, v=ones, confidence=ones)
