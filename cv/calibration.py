"""Calibration du bassin : homographie image (px) → bassin (m).

L'utilisateur clique au moins 4 repères sur une image (coins, croisements des
lignes d'eau avec les murs, marques 5 m / 15 m) et saisit leurs coordonnées réelles :
`x` le long du bassin depuis le mur de départ, `y` transversal, `y = 0` au bord
extérieur du couloir `first_lane`. Les couloirs font `lane_width` de large.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from cv.domain import FloatArray

MAX_REPROJECTION_ERROR_M = 0.3
DEFAULT_LANE_WIDTH_M = 2.5


class CalibrationError(ValueError):
    """Repères insuffisants, dégénérés ou trop imprécis."""


@dataclass(frozen=True)
class Calibration:
    image_points: FloatArray  # px, (n, 2)
    pool_points: FloatArray  # m, (n, 2)
    homography: FloatArray  # 3×3, px → m
    point_errors: FloatArray  # m, erreur de reprojection de chaque repère
    lane_width: float = DEFAULT_LANE_WIDTH_M
    first_lane: int = 1
    reprojection_error: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reprojection_error", float(self.point_errors.max()))

    def lane_band(self, lane: int) -> tuple[float, float]:
        """(y_min, y_max) du couloir en mètres."""
        offset = lane - self.first_lane
        if offset < 0:
            raise ValueError(f"couloir {lane} avant le premier couloir ({self.first_lane})")
        return offset * self.lane_width, (offset + 1) * self.lane_width


def transform(homography: FloatArray, points: FloatArray) -> FloatArray:
    """Applique une homographie à des points (n, 2)."""
    projected = cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography)
    return np.asarray(projected, dtype=np.float64).reshape(-1, 2)


def calibrate(
    image_points: FloatArray,
    pool_points: FloatArray,
    lane_width: float = DEFAULT_LANE_WIDTH_M,
    first_lane: int = 1,
) -> Calibration:
    """Homographie par RANSAC ; refusée si un repère est à plus de 0,3 m.

    Avec exactement 4 repères, l'homographie passe par tous : l'erreur est nulle et ne
    permet pas de détecter un clic faux. En demander 6 ou plus dans l'interface.
    """
    image = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    pool = np.asarray(pool_points, dtype=np.float64).reshape(-1, 2)
    if image.shape != pool.shape:
        raise CalibrationError("autant de coordonnées réelles que de repères cliqués")
    if len(image) < 4:
        raise CalibrationError("au moins 4 repères sont nécessaires")

    fitted, _ = cv2.findHomography(image, pool, cv2.RANSAC, MAX_REPROJECTION_ERROR_M)
    if fitted is None:
        raise CalibrationError("repères dégénérés (trois repères alignés ?)")
    homography: FloatArray = np.asarray(fitted, dtype=np.float64)
    errors = np.linalg.norm(transform(homography, image) - pool, axis=1)
    worst = int(np.argmax(errors))
    if errors[worst] > MAX_REPROJECTION_ERROR_M:
        raise CalibrationError(
            f"repère {worst + 1} à {errors[worst]:.2f} m de sa position réelle "
            f"(maximum {MAX_REPROJECTION_ERROR_M} m) : vérifier le clic ou ses coordonnées"
        )
    return Calibration(image, pool, homography, errors, lane_width, first_lane)


def save_calibration(calibration: Calibration, path: Path) -> None:
    data = {
        "image_points": calibration.image_points.tolist(),
        "pool_points": calibration.pool_points.tolist(),
        "lane_width": calibration.lane_width,
        "first_lane": calibration.first_lane,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_calibration(path: Path) -> Calibration:
    """Relit les repères et recalcule (et revalide) l'homographie."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return calibrate(
        np.array(data["image_points"], dtype=np.float64),
        np.array(data["pool_points"], dtype=np.float64),
        lane_width=float(data.get("lane_width", DEFAULT_LANE_WIDTH_M)),
        first_lane=int(data.get("first_lane", 1)),
    )
