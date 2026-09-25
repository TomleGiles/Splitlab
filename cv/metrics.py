"""Métriques de course calculées à partir de la trajectoire et des événements.

Fonctions pures : pas d'I/O, pas d'OpenCV. Définitions de référence : CLAUDE.md.

Chaque métrique porte une confiance (minimum des confiances des événements et
des échantillons de trajectoire utilisés) et le flag `manually_corrected`
(vrai si au moins un événement utilisé a été corrigé à la main).
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from cv.domain import (
    NOT_MEASURABLE,
    POOL_LENGTH,
    RACE_DISTANCE,
    Event,
    EventType,
    FloatArray,
    MetricValue,
    Trajectory,
)

SPLIT_DISTANCES = (15.0, 25.0, 35.0, 50.0)  # m, distance parcourue
TURN_WINDOW = 5.0  # m avant et après le mur
FINISH_WINDOW = 5.0  # m
PROFILE_STEP = 0.5  # m

_UNIQUE_EVENTS = (
    EventType.START_SIGNAL,
    EventType.ENTRY,
    EventType.WALL_IN,
    EventType.WALL_OUT,
    EventType.FINISH,
)


@dataclass(frozen=True, slots=True)
class SectionMetrics:
    """Métriques de nage d'une section de nage libre (une longueur)."""

    stroke_rate: MetricValue  # cycles/min
    stroke_length: MetricValue  # m/cycle
    stroke_index: MetricValue  # m²/s


@dataclass(frozen=True, slots=True)
class VelocityProfile:
    """v(d) échantillonnée tous les `PROFILE_STEP` m ; NaN là où d n'est pas observée."""

    d: FloatArray  # m
    v: FloatArray  # m/s
    confidence: FloatArray


@dataclass(frozen=True, slots=True)
class RaceMetrics:
    reaction_time: MetricValue
    splits: dict[float, MetricValue]  # distance parcourue (m) → temps (s)
    underwater_start: MetricValue  # m, depuis le mur de départ
    underwater_turn: MetricValue  # m, depuis le mur de virage
    sections: tuple[SectionMetrics, SectionMetrics]  # aller, retour
    turn_time: MetricValue
    finish_speed: MetricValue
    velocity_profile: VelocityProfile


_SECTION_NOT_MEASURABLE = SectionMetrics(NOT_MEASURABLE, NOT_MEASURABLE, NOT_MEASURABLE)


def _measured(
    value: float, confidences: Iterable[float] = (), events: Iterable[Event] = ()
) -> MetricValue:
    events = list(events)
    all_confidences = [*confidences, *(e.confidence for e in events)]
    return MetricValue(
        value=float(value),
        confidence=float(min(all_confidences, default=1.0)),
        manually_corrected=any(e.manually_corrected for e in events),
    )


def _first_crossings(
    traj: Trajectory, targets: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Première traversée de chaque distance cible par d(t).

    Retourne (t, v, confiance) interpolés linéairement ; NaN (confiance 0) pour
    les distances non observées (avant le premier échantillon ou jamais atteintes).
    """
    d = traj.d
    # d(t) n'est pas forcément monotone (bruit) : la première traversée de `target`
    # est le premier indice où le maximum courant atteint `target`.
    i = np.searchsorted(np.maximum.accumulate(d), targets, side="left")
    t_out = np.full(targets.shape, np.nan)
    v_out = np.full(targets.shape, np.nan)
    c_out = np.zeros(targets.shape)

    at_start = (i == 0) & (targets == d[0])
    t_out[at_start] = traj.t[0]
    v_out[at_start] = traj.v[0]
    c_out[at_start] = traj.confidence[0]

    inside = (i > 0) & (i < d.size)
    hi = i[inside]
    lo = hi - 1
    # d[lo] < target <= d[hi] par construction, donc d[hi] > d[lo].
    frac = (targets[inside] - d[lo]) / (d[hi] - d[lo])
    t_out[inside] = traj.t[lo] + frac * (traj.t[hi] - traj.t[lo])
    v_out[inside] = traj.v[lo] + frac * (traj.v[hi] - traj.v[lo])
    c_out[inside] = np.minimum(traj.confidence[lo], traj.confidence[hi])
    return t_out, v_out, c_out


def _time_at_distance(traj: Trajectory, distance: float) -> tuple[float, float] | None:
    """(t, confiance) de la première traversée de `distance`, ou None si non observée."""
    t, _, c = _first_crossings(traj, np.array([distance]))
    if np.isnan(t[0]):
        return None
    return float(t[0]), float(c[0])


def _value_at_time(traj: Trajectory, values: FloatArray, t: float) -> tuple[float, float] | None:
    """(valeur interpolée, confiance) à l'instant t, ou None hors de la trajectoire."""
    if not traj.t[0] <= t <= traj.t[-1]:
        return None
    i = int(np.searchsorted(traj.t, t, side="left"))
    lo, hi = max(i - 1, 0), min(i, traj.t.size - 1)
    confidence = min(traj.confidence[lo], traj.confidence[hi])
    return float(np.interp(t, traj.t, values)), float(confidence)


def _unique_event(events: Sequence[Event], event_type: EventType) -> Event | None:
    matches = [e for e in events if e.type == event_type]
    if len(matches) > 1:
        raise ValueError(f"plusieurs événements {event_type} : un seul attendu")
    return matches[0] if matches else None


def _breakouts(
    events: Sequence[Event], wall_in: Event | None, wall_out: Event | None
) -> tuple[Event | None, Event | None]:
    """Coulée après le départ (avant `wall_in`) et après le virage (après `wall_out`)."""
    breakouts = [e for e in events if e.type == EventType.BREAKOUT]
    before = [e for e in breakouts if wall_in is None or e.t < wall_in.t]
    after = [e for e in breakouts if wall_out is not None and e.t > wall_out.t]
    if len(before) > 1 or len(after) > 1:
        raise ValueError("au plus une reprise de nage attendue par longueur")
    if len(before) + len(after) != len(breakouts):
        raise ValueError("reprise de nage pendant le virage")
    return (before[0] if before else None), (after[0] if after else None)


def splits(
    traj: Trajectory, wall_in: Event | None, finish: Event | None
) -> dict[float, MetricValue]:
    """Temps de passage aux distances `SPLIT_DISTANCES` (distance parcourue).

    Aux murs (25 m, 50 m), le point suivi n'atteint jamais la distance : le passage
    est l'événement de contact (`wall_in`, `finish`).
    """
    wall_events = {POOL_LENGTH: wall_in, RACE_DISTANCE: finish}
    result: dict[float, MetricValue] = {}
    for distance in SPLIT_DISTANCES:
        if distance in wall_events:
            event = wall_events[distance]
            result[distance] = (
                NOT_MEASURABLE if event is None else _measured(event.t, events=[event])
            )
        else:
            crossing = _time_at_distance(traj, distance)
            result[distance] = (
                NOT_MEASURABLE if crossing is None else _measured(crossing[0], [crossing[1]])
            )
    return result


def underwater_distance(
    traj: Trajectory, breakout: Event | None, from_wall_x: float
) -> MetricValue:
    """Distance de coulée : `x` au `breakout`, mesurée depuis le mur quitté (x = 0 ou 25 m)."""
    if breakout is None:
        return NOT_MEASURABLE
    x = _value_at_time(traj, traj.x, breakout.t)
    if x is None:
        return NOT_MEASURABLE
    return _measured(abs(x[0] - from_wall_x), [x[1]], [breakout])


def stroke_section(traj: Trajectory, cycle_starts: Sequence[Event]) -> SectionMetrics:
    """SR, SL et SI d'une section à partir des débuts de cycles successifs.

    n marqueurs délimitent n − 1 cycles complets. La vitesse moyenne est prise sur
    la même fenêtre (premier → dernier marqueur), donc SL = distance / nombre de cycles.
    """
    if len(cycle_starts) < 2:
        return _SECTION_NOT_MEASURABLE
    first, last = cycle_starts[0], cycle_starts[-1]
    duration = last.t - first.t
    if any(b.t <= a.t for a, b in pairwise(cycle_starts)):
        raise ValueError("les débuts de cycle doivent être strictement croissants")

    stroke_rate = 60.0 * (len(cycle_starts) - 1) / duration
    sr = _measured(stroke_rate, events=cycle_starts)
    d_first = _value_at_time(traj, traj.d, first.t)
    d_last = _value_at_time(traj, traj.d, last.t)
    if d_first is None or d_last is None:
        return SectionMetrics(sr, NOT_MEASURABLE, NOT_MEASURABLE)

    speed = (d_last[0] - d_first[0]) / duration
    stroke_length = speed / (stroke_rate / 60.0)
    confidences = [d_first[1], d_last[1]]
    return SectionMetrics(
        stroke_rate=sr,
        stroke_length=_measured(stroke_length, confidences, cycle_starts),
        stroke_index=_measured(speed * stroke_length, confidences, cycle_starts),
    )


def turn_time(traj: Trajectory) -> MetricValue:
    """Temps entre 5 m avant et 5 m après le mur (d = 20 → 30 m)."""
    before = _time_at_distance(traj, POOL_LENGTH - TURN_WINDOW)
    after = _time_at_distance(traj, POOL_LENGTH + TURN_WINDOW)
    if before is None or after is None:
        return NOT_MEASURABLE
    return _measured(after[0] - before[0], [before[1], after[1]])


def finish_speed(traj: Trajectory, finish: Event | None) -> MetricValue:
    """Vitesse moyenne sur les 5 derniers mètres (d = 45 m → contact d'arrivée)."""
    if finish is None:
        return NOT_MEASURABLE
    start = _time_at_distance(traj, RACE_DISTANCE - FINISH_WINDOW)
    if start is None or finish.t <= start[0]:
        return NOT_MEASURABLE
    return _measured(FINISH_WINDOW / (finish.t - start[0]), [start[1]], [finish])


def velocity_profile(traj: Trajectory, step: float = PROFILE_STEP) -> VelocityProfile:
    """v(d) de 0 à 50 m tous les `step` m, lue à la première traversée de chaque distance."""
    grid = np.arange(0.0, RACE_DISTANCE + step / 2, step)
    _, v, confidence = _first_crossings(traj, grid)
    return VelocityProfile(d=grid, v=v, confidence=confidence)


def compute_race_metrics(traj: Trajectory, events: Sequence[Event]) -> RaceMetrics:
    events = sorted(events, key=lambda e: e.t)
    unique = {event_type: _unique_event(events, event_type) for event_type in _UNIQUE_EVENTS}
    wall_in, wall_out = unique[EventType.WALL_IN], unique[EventType.WALL_OUT]
    finish = unique[EventType.FINISH]
    breakout_start, breakout_turn = _breakouts(events, wall_in, wall_out)

    cycles = [e for e in events if e.type == EventType.STROKE_CYCLE]
    outbound = [e for e in cycles if wall_in is not None and e.t < wall_in.t]
    inbound = [e for e in cycles if wall_out is not None and e.t > wall_out.t]

    return RaceMetrics(
        # Le dernier contact plot n'est pas encore un événement du modèle :
        # temps de réaction non mesurable, jamais estimé.
        reaction_time=NOT_MEASURABLE,
        splits=splits(traj, wall_in, finish),
        underwater_start=underwater_distance(traj, breakout_start, from_wall_x=0.0),
        underwater_turn=underwater_distance(traj, breakout_turn, from_wall_x=POOL_LENGTH),
        sections=(stroke_section(traj, outbound), stroke_section(traj, inbound)),
        turn_time=turn_time(traj),
        finish_speed=finish_speed(traj, finish),
        velocity_profile=velocity_profile(traj),
    )
