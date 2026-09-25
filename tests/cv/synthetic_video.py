"""Vidéo synthétique : bassin vu de biais depuis les gradins, nageurs en rectangles.

Le bassin est dessiné vu de dessus (`CANVAS_PPM` px/m), puis projeté dans l'image
caméra par une homographie connue : la vérité terrain est exacte.
"""

from collections.abc import Callable, Iterator

import cv2
import numpy as np
from numpy.typing import NDArray

from cv.calibration import Calibration, calibrate, transform
from cv.domain import POOL_LENGTH, FloatArray

CANVAS_PPM = 40
LANE_WIDTH = 2.5
LANES = 8
POOL_WIDTH = LANES * LANE_WIDTH
FRAME_SIZE = (960, 540)
FPS = 25.0

# Coins du bassin (m) → image (px) : côté opposé en haut, plus étroit (vue de biais).
POOL_CORNERS = np.array([[0, 0], [POOL_LENGTH, 0], [POOL_LENGTH, POOL_WIDTH], [0, POOL_WIDTH]])
IMAGE_CORNERS = np.array([[250, 150], [710, 150], [940, 520], [20, 520]], dtype=np.float64)
POOL_TO_IMAGE = np.asarray(
    cv2.getPerspectiveTransform(POOL_CORNERS.astype(np.float32), IMAGE_CORNERS.astype(np.float32)),
    dtype=np.float64,
)

WATER_BGR = (208, 200, 64)
ROPE_BGR = (40, 40, 200)
LANE_LINE_BGR = (120, 110, 30)
BODY_BGR = (235, 235, 235)  # écume et peau
HEAD_BGR = (40, 40, 40)  # bonnet
STANDS_BGR = (80, 80, 80)

BODY_LENGTH = 1.8  # m
BODY_WIDTH = 0.5  # m

Swimmer = tuple[int, Callable[[float], float | None]]  # couloir, t → avant du nageur (m)


def _px(meters: float) -> int:
    return round(meters * CANVAS_PPM)


def empty_pool(rng: np.random.Generator) -> NDArray[np.uint8]:
    height, width = _px(POOL_WIDTH), _px(POOL_LENGTH)
    canvas = np.full((height, width, 3), WATER_BGR, dtype=np.float64)
    canvas += rng.normal(0.0, 4.0, canvas.shape)  # texture de l'eau
    pool = np.clip(canvas, 0, 255).astype(np.uint8)
    for k in range(LANES):
        y = _px((k + 0.5) * LANE_WIDTH)
        cv2.line(pool, (_px(2.0), y), (_px(POOL_LENGTH - 2.0), y), LANE_LINE_BGR, _px(0.2))
    for k in range(LANES + 1):
        y = _px(k * LANE_WIDTH)
        cv2.line(pool, (0, y), (_px(POOL_LENGTH), y), ROPE_BGR, _px(0.1))
    return pool


def draw_swimmer(pool: NDArray[np.uint8], lane: int, front: float) -> None:
    y = (lane - 0.5) * LANE_WIDTH  # couloir 1 : y de 0 à 2,5 m
    top_left = (_px(front - BODY_LENGTH), _px(y - BODY_WIDTH / 2))
    bottom_right = (_px(front), _px(y + BODY_WIDTH / 2))
    cv2.rectangle(pool, top_left, bottom_right, BODY_BGR, thickness=-1)
    cv2.circle(pool, (_px(front - 0.35), _px(y)), _px(0.12), HEAD_BGR, thickness=-1)


def render(swimmers: list[Swimmer], duration: float, seed: int = 0) -> Iterator[NDArray[np.uint8]]:
    rng = np.random.default_rng(seed)
    pool = empty_pool(rng)
    canvas_to_image = POOL_TO_IMAGE @ np.diag([1 / CANVAS_PPM, 1 / CANVAS_PPM, 1.0])
    for i in range(round(duration * FPS)):
        t = i / FPS
        frame_pool = pool.copy()
        for lane, front_at in swimmers:
            front = front_at(t)
            if front is not None:
                draw_swimmer(frame_pool, lane, front)
        frame = cv2.warpPerspective(frame_pool, canvas_to_image, FRAME_SIZE, borderValue=STANDS_BGR)
        noise = rng.normal(0.0, 3.0, frame.shape)
        yield np.clip(frame + noise, 0, 255).astype(np.uint8)


def calibration_points() -> tuple[FloatArray, FloatArray]:
    """Six repères : quatre coins et deux marques sur les lignes d'eau."""
    pool = np.array([*POOL_CORNERS, [5.0, 5.0], [15.0, 12.5]], dtype=np.float64)
    return transform(POOL_TO_IMAGE, pool), pool


def synthetic_calibration() -> Calibration:
    image, pool = calibration_points()
    return calibrate(image, pool, lane_width=LANE_WIDTH, first_lane=1)
