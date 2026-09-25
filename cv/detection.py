"""Détection du nageur dans son couloir, sans apprentissage.

1. Le cœur du couloir cible est redressé par homographie en une bande : une colonne
   par pas de `1 / PIXELS_PER_M` m le long du bassin.
2. Le fond (eau, lignes, cordes) est la médiane temporelle des bandes : le nageur ne
   reste jamais longtemps au même endroit.
3. Le nageur est la zone qui s'écarte du fond (écume, tête, corps) ; on garde la plus
   grosse zone continue de colonnes occupées.

Seuls les points au niveau de l'eau sont bien placés (plan de l'homographie) : sur
le plot et pendant le vol du plongeon, la position est approximative.
Choix de cette approche : docs/essais/2026-09-25-detection-yolo.md.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter

from cv.calibration import Calibration
from cv.domain import POOL_LENGTH, FloatArray

Image = NDArray[np.uint8]

PIXELS_PER_M = 20
X_MARGIN_M = 0.5  # bande analysée un peu au-delà des murs
LANE_CORE = 0.6  # fraction centrale du couloir analysée : évite cordes et voisins
BACKGROUND_SAMPLES = 60
COLOR_THRESHOLD = 30.0  # distance au fond dans l'espace Lab (0–255)
COLUMN_THRESHOLD = 0.15  # part de la hauteur de bande occupée pour qu'une colonne compte
MERGE_GAP_M = 0.5  # zones séparées de moins : même nageur (bras, jambes, écume)
MIN_BLOB_M = 0.4  # zone plus courte : reflet ou vaguelette
TURN_SMOOTHING = 9  # images, médiane glissante pour repérer le demi-tour


@dataclass(frozen=True, slots=True)
class Observation:
    """Étendue du nageur le long du bassin (m) dans une image."""

    x_min: float
    x_max: float
    x_center: float  # centre de masse de la zone
    confidence: float  # part de la zone retenue dans tout ce qui diffère du fond


@dataclass(frozen=True, slots=True)
class LaneDetections:
    """Une valeur par image ; NaN (confiance 0) quand le nageur n'est pas vu."""

    x_min: FloatArray
    x_max: FloatArray
    x_center: FloatArray
    confidence: FloatArray


@dataclass(frozen=True, slots=True)
class LaneStrip:
    """Transformation image → bande redressée du cœur d'un couloir."""

    matrix: FloatArray  # 3×3, px image → px bande
    size: tuple[int, int]  # (largeur, hauteur) en px

    @classmethod
    def for_lane(cls, calibration: Calibration, lane: int) -> "LaneStrip":
        y_min, y_max = calibration.lane_band(lane)
        center, half = (y_min + y_max) / 2, LANE_CORE * (y_max - y_min) / 2
        x0 = -X_MARGIN_M
        pool_to_strip = np.array(
            [
                [PIXELS_PER_M, 0.0, -x0 * PIXELS_PER_M],
                [0.0, PIXELS_PER_M, -(center - half) * PIXELS_PER_M],
                [0.0, 0.0, 1.0],
            ]
        )
        width = round((POOL_LENGTH + 2 * X_MARGIN_M) * PIXELS_PER_M)
        height = max(1, round(2 * half * PIXELS_PER_M))
        return cls(pool_to_strip @ calibration.homography, (width, height))

    def rectify(self, frame: Image) -> Image:
        strip = cv2.warpPerspective(frame, self.matrix, self.size, flags=cv2.INTER_LINEAR)
        return np.asarray(strip, dtype=np.uint8)


def column_to_x(column: float) -> float:
    return column / PIXELS_PER_M - X_MARGIN_M


def background(strips: list[Image]) -> Image:
    """Médiane temporelle d'au plus `BACKGROUND_SAMPLES` bandes réparties dans la vidéo."""
    step = max(1, len(strips) // BACKGROUND_SAMPLES)
    return np.asarray(np.median(np.stack(strips[::step]), axis=0), dtype=np.uint8)


def foreground(strip: Image, background_lab: NDArray[np.float32]) -> NDArray[np.bool_]:
    lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB).astype(np.float32)
    mask = (np.linalg.norm(lab - background_lab, axis=2) > COLOR_THRESHOLD).astype(np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return np.asarray(opened, dtype=bool)


def _runs(occupied: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Plages [début, fin) de colonnes occupées, fusionnées si l'écart est petit."""
    edges = np.diff(np.concatenate(([0], occupied.astype(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    runs: list[tuple[int, int]] = []
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        if runs and start - runs[-1][1] <= MERGE_GAP_M * PIXELS_PER_M:
            runs[-1] = (runs[-1][0], end)
        else:
            runs.append((start, end))
    return runs


def locate(mask: NDArray[np.bool_]) -> Observation | None:
    """Plus grosse zone continue de colonnes occupées, ou None."""
    occupancy = mask.mean(axis=0)
    runs = _runs(occupancy > COLUMN_THRESHOLD)
    if not runs:
        return None
    masses = [float(occupancy[a:b].sum()) for a, b in runs]
    best = int(np.argmax(masses))
    start, end = runs[best]
    if end - start < MIN_BLOB_M * PIXELS_PER_M:
        return None
    columns = np.arange(start, end)
    center = float(np.average(columns, weights=occupancy[start:end]))
    return Observation(
        x_min=column_to_x(start),
        x_max=column_to_x(end),
        x_center=column_to_x(center),
        confidence=masses[best] / sum(masses),
    )


def detect_frames(frames: Iterable[Image], strip: LaneStrip) -> LaneDetections:
    strips = [strip.rectify(frame) for frame in frames]
    if not strips:
        raise ValueError("aucune image")
    background_lab = cv2.cvtColor(background(strips), cv2.COLOR_BGR2LAB).astype(np.float32)
    observations = [locate(foreground(s, background_lab)) for s in strips]

    def column(attribute: str) -> FloatArray:
        return np.array(
            [np.nan if o is None else getattr(o, attribute) for o in observations],
            dtype=np.float64,
        )

    return LaneDetections(
        x_min=column("x_min"),
        x_max=column("x_max"),
        x_center=column("x_center"),
        confidence=np.nan_to_num(column("confidence"), nan=0.0),
    )


def _read_frames(capture: cv2.VideoCapture) -> Iterator[Image]:
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        yield np.asarray(frame, dtype=np.uint8)


def detect_video(path: Path, calibration: Calibration, lane: int) -> tuple[LaneDetections, float]:
    """Détections image par image et fréquence d'images de la vidéo."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"vidéo illisible : {path.name}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        detections = detect_frames(_read_frames(capture), LaneStrip.for_lane(calibration, lane))
    finally:
        capture.release()
    return detections, fps


def front_position(detections: LaneDetections) -> FloatArray:
    """Avant du nageur : `x_max` à l'aller, `x_min` au retour.

    Le demi-tour est l'image où le centre de la zone (lissé) est le plus loin du départ.
    """
    seen = np.flatnonzero(~np.isnan(detections.x_center))
    if seen.size == 0:
        return np.full_like(detections.x_center, np.nan)
    smoothed = median_filter(detections.x_center[seen], size=TURN_SMOOTHING, mode="nearest")
    # La médiane aplatit le sommet : on prend le milieu du plateau, pas son début.
    plateau = np.flatnonzero(smoothed == smoothed.max())
    turn = seen[int(plateau[plateau.size // 2])]
    index = np.arange(detections.x_center.size)
    return np.where(index <= turn, detections.x_max, detections.x_min)
