"""Types du domaine partagés par le pipeline : événements, trajectoire, valeurs de métriques.

Conventions (voir CLAUDE.md) :
- unités SI : secondes, mètres, m/s ;
- `t = 0` au signal de départ ;
- `x` = distance au mur de départ (0 → 25 m, redescend après le virage) ;
- `d` = distance parcourue (0 → 50 m).
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class EventType(StrEnum):
    START_SIGNAL = "start_signal"
    ENTRY = "entry"
    BREAKOUT = "breakout"
    STROKE_CYCLE = "stroke_cycle"
    WALL_IN = "wall_in"
    WALL_OUT = "wall_out"
    FINISH = "finish"


def _check_confidence(confidence: float) -> None:
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confiance hors de [0, 1] : {confidence}")


@dataclass(frozen=True, slots=True)
class Event:
    """Événement de course. Un `stroke_cycle` marque le début d'un cycle (deux bras)."""

    type: EventType
    t: float  # s, t = 0 au signal de départ
    frame: int
    confidence: float
    manually_corrected: bool = False

    def __post_init__(self) -> None:
        _check_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Trajectoire lissée du nageur, échantillonnée aux instants `t` (strictement croissants)."""

    t: FloatArray  # s
    x: FloatArray  # m, repère bassin
    d: FloatArray  # m, distance parcourue
    v: FloatArray  # m/s
    confidence: FloatArray  # 0–1 par échantillon

    def __post_init__(self) -> None:
        arrays = (self.t, self.x, self.d, self.v, self.confidence)
        if any(a.ndim != 1 or a.shape != self.t.shape for a in arrays):
            raise ValueError("les tableaux de la trajectoire doivent être 1D et de même taille")
        if self.t.size < 2:
            raise ValueError("une trajectoire contient au moins deux échantillons")
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("les instants t doivent être strictement croissants")
        if np.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("confiance hors de [0, 1]")


@dataclass(frozen=True, slots=True)
class MetricValue:
    """Valeur de métrique. `value is None` : non mesurable (jamais estimée)."""

    value: float | None
    confidence: float
    manually_corrected: bool = False

    def __post_init__(self) -> None:
        _check_confidence(self.confidence)


NOT_MEASURABLE = MetricValue(value=None, confidence=0.0)
