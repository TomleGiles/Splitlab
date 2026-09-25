"""Course de démonstration : un 50 NL simulé, passé dans le vrai calcul.

Aucune vidéo : les positions brutes du nageur et les événements sont **simulés**
(données synthétiques, affichées comme telles dans le front). Sert à montrer la
fiche de course tant que la détection vidéo n'existe pas.

La simulation suit la tête du nageur : départ, coulée, nage avec oscillation
intra-cycle et fatigue, demi-tour ~1,4 m avant le mur, coulée retour, arrivée.
"""

import math

import numpy as np

from cv.domain import POOL_LENGTH, Event, EventType, Trajectory
from cv.trajectory import build_trajectory

FPS = 50.0
DT = 1.0 / FPS

HEAD_X_ON_BLOCK = 0.5  # m
# Vitesse de la tête (m/s) du signal à la reprise de nage : plot, vol, coulée.
START_PROFILE = ((0.0, 0.0), (0.40, 0.3), (0.70, 4.0), (1.05, 3.7), (1.7, 2.6), (2.7, 2.2))
BLOCK_OFF_T = 0.70
ENTRY_T = 1.05
BREAKOUT_START_T = 4.0
BREAKOUT_START_SPEED = 2.0

FLIP_START_X = 22.3  # m, la tête commence à ralentir pour la culbute
FLIP_DURATION = 0.45  # s, jusqu'au demi-tour de la tête
WALL_IN_AFTER_APEX = 0.20  # s, contact des pieds
WALL_OUT_AFTER_APEX = 0.45  # s, fin de poussée
PUSH_OFF_SPEED = 2.7
GLIDE_SPEED = 1.95
GLIDE_TIME_CONSTANT = 1.0  # s
TURN_UNDERWATER_DISTANCE = 8.0  # m depuis le mur
FINISH_HEAD_X = 0.7  # m : la main touche, bras tendu

# (vitesse de départ m/s, perte m/s par seconde, fréquence cycles/min) par longueur :
# fréquence tenue au retour, vitesse en baisse → amplitude en baisse (fatigue).
OUTBOUND_SWIM = (1.92, 0.012, 52.0)
INBOUND_SWIM = (1.84, 0.016, 51.5)
INTRA_CYCLE_AMPLITUDE = 0.10  # m/s, deux pics par cycle (un par bras)

DETECTION_NOISE_M = 0.04
SURFACE_CONFIDENCE = 0.9
UNDERWATER_CONFIDENCE = 0.78
FLIP_SPLASH_S = 0.3  # détection perdue autour du demi-tour


class _Head:
    """Distance parcourue par la tête le long de son chemin, échantillonnée à FPS."""

    def __init__(self, start: float) -> None:
        self.path = [start]

    @property
    def now(self) -> float:
        return (len(self.path) - 1) * DT

    @property
    def position(self) -> float:
        return self.path[-1]

    def move(self, speed: float) -> None:
        self.path.append(self.path[-1] + speed * DT)


def _stroking_speed(t: float, since: float, swim: tuple[float, float, float]) -> float:
    base, loss, stroke_rate = swim
    phase = 2.0 * (t - since) * stroke_rate / 60.0  # deux coups de bras par cycle
    return base - loss * (t - since) + INTRA_CYCLE_AMPLITUDE * math.sin(2 * math.pi * phase)


def _cycle_markers(first: float, last: float, stroke_rate: float, low: set[int]) -> list[Event]:
    period = 60.0 / stroke_rate
    times = np.arange(first, last, period)
    return [
        _event(EventType.STROKE_CYCLE, t, 0.6 if k in low else 0.85) for k, t in enumerate(times)
    ]


def _event(event_type: EventType, t: float, confidence: float) -> Event:
    return Event(event_type, float(t), frame=round(t * FPS), confidence=confidence)


def simulate_demo_race(seed: int = 7) -> tuple[Trajectory, list[Event]]:
    """Trajectoire (issue de `build_trajectory`) et événements d'un 50 NL simulé."""
    head = _Head(HEAD_X_ON_BLOCK)
    start_t, start_v = zip(*START_PROFILE, strict=True)
    while head.now < BREAKOUT_START_T:
        speed = float(
            np.interp(head.now, (*start_t, BREAKOUT_START_T), (*start_v, BREAKOUT_START_SPEED))
        )
        head.move(speed)

    while head.position < FLIP_START_X:
        head.move(_stroking_speed(head.now, BREAKOUT_START_T, OUTBOUND_SWIM))
    flip_start_t = head.now
    approach_speed = _stroking_speed(head.now, BREAKOUT_START_T, OUTBOUND_SWIM)
    flip_steps = round(FLIP_DURATION / DT)
    for i in range(1, flip_steps + 1):
        head.move(approach_speed * (1 - i / flip_steps))
    apex_t, apex_path = head.now, head.position

    push_steps = round((WALL_OUT_AFTER_APEX + 0.1) / DT)
    for i in range(1, push_steps + 1):
        head.move(PUSH_OFF_SPEED * i / push_steps)
    push_end_t = head.now
    # x = 2·apex − chemin ; coulée finie quand 25 − x atteint TURN_UNDERWATER_DISTANCE.
    breakout_path = 2 * apex_path - (POOL_LENGTH - TURN_UNDERWATER_DISTANCE)
    while head.position < breakout_path:
        decay = np.exp(-(head.now - push_end_t) / GLIDE_TIME_CONSTANT)
        head.move(GLIDE_SPEED + (PUSH_OFF_SPEED - GLIDE_SPEED) * decay)
    breakout_turn_t = head.now

    finish_path = 2 * apex_path - FINISH_HEAD_X
    while head.position < finish_path:
        head.move(_stroking_speed(head.now, breakout_turn_t, INBOUND_SWIM))
    finish_t = head.now

    events = [
        _event(EventType.START_SIGNAL, 0.0, 1.0),
        _event(EventType.BLOCK_OFF, BLOCK_OFF_T, 0.9),
        _event(EventType.ENTRY, ENTRY_T, 0.85),
        _event(EventType.BREAKOUT, BREAKOUT_START_T, 0.8),
        *_cycle_markers(BREAKOUT_START_T + 0.2, flip_start_t - 0.2, OUTBOUND_SWIM[2], low=set()),
        _event(EventType.WALL_IN, apex_t + WALL_IN_AFTER_APEX, 0.85),
        _event(EventType.WALL_OUT, apex_t + WALL_OUT_AFTER_APEX, 0.85),
        _event(EventType.BREAKOUT, breakout_turn_t, 0.8),
        # Éclaboussures sur le retour : deux marqueurs incertains.
        *_cycle_markers(breakout_turn_t + 0.2, finish_t - 0.3, INBOUND_SWIM[2], low={3, 9}),
        _event(EventType.FINISH, finish_t, 0.95),
    ]

    path = np.array(head.path)
    t = np.arange(path.size, dtype=np.float64) * DT
    x_true = np.where(path <= apex_path, path, 2 * apex_path - path)
    rng = np.random.default_rng(seed)
    x = x_true + rng.normal(0.0, DETECTION_NOISE_M, t.size)
    confidence = np.full(t.size, SURFACE_CONFIDENCE)
    underwater = ((t > ENTRY_T) & (t < BREAKOUT_START_T)) | (
        (t > apex_t + WALL_OUT_AFTER_APEX) & (t < breakout_turn_t)
    )
    confidence[underwater] = UNDERWATER_CONFIDENCE
    x[np.abs(t - apex_t) < FLIP_SPLASH_S / 2] = np.nan

    return build_trajectory(t, x, confidence), events
