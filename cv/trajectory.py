"""Trajectoire du nageur : nettoyage, lissage, x(t), d(t) et v(t) en repère bassin.

Entrée : position brute `x` (m, repère bassin, après homographie) à chaque frame,
avec la confiance de détection. Sortie : une `Trajectory` sans trou.

Le lissage (Savitzky-Golay, qui donne aussi la dérivée) est fait séparément sur
chaque longueur : jamais à travers le demi-tour, où la vitesse change de signe.
Le point suivi fait demi-tour avant le mur, donc `d` saute au virage
(voir docs/questions-ouvertes.md).
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from cv.domain import POOL_LENGTH, FloatArray, Trajectory

BoolArray = NDArray[np.bool_]

SMOOTHING_WINDOW_S = 1.0  # ~ un cycle de bras : gomme les oscillations intra-cycle
SMOOTHING_POLYORDER = 2
OUTLIER_WINDOW_S = 0.5
OUTLIER_THRESHOLD_M = 1.0  # écart à la médiane glissante au-delà duquel x est rejeté
MAX_INTERPOLATED_GAP_S = 0.5  # trou plus long : interpolé mais confiance 0
INTERPOLATED_CONFIDENCE_FACTOR = 0.5
EDGE_CONFIDENCE_FACTOR = 0.5  # demi-fenêtre au début et à la fin de chaque longueur


def build_trajectory(
    t: FloatArray,
    x: FloatArray,
    confidence: FloatArray,
    turn_t: float | None = None,
) -> Trajectory:
    """Construit la trajectoire lissée à partir des positions brutes, une par frame.

    t : instants des frames (s, t = 0 au signal de départ), à pas constant.
    x : position brute (m), NaN quand le nageur n'est pas détecté.
    confidence : confiance de détection (0–1).
    turn_t : instant du demi-tour s'il est connu (ex. après correction manuelle) ;
        sinon, instant où x lissé est maximal.
    """
    if not (t.ndim == 1 and t.shape == x.shape == confidence.shape):
        raise ValueError("t, x et confidence doivent être 1D et de même taille")
    dt = _frame_period(t)

    valid = np.isfinite(x) & (confidence > 0)
    valid &= ~_outliers(x, valid, _odd_window(OUTLIER_WINDOW_S, dt))
    if np.count_nonzero(valid) < 2:
        raise ValueError("moins de deux détections exploitables")

    # Avant la première et après la dernière détection : non observé, pas extrapolé.
    first, last = np.flatnonzero(valid)[[0, -1]]
    kept = slice(first, last + 1)
    t, x, confidence, valid = t[kept], x[kept], confidence[kept], valid[kept]
    x_filled, confidence = _fill_gaps(t, x, confidence, valid, dt)

    window = _odd_window(SMOOTHING_WINDOW_S, dt)
    if turn_t is None:
        turn_t = float(t[np.argmax(_smooth(x_filled, window, dt)[0])])
    outbound = t <= turn_t

    x_smooth = np.empty_like(x_filled)
    dx_dt = np.empty_like(x_filled)
    for length in (outbound, ~outbound):
        if length.any():
            x_smooth[length], dx_dt[length] = _smooth(x_filled[length], window, dt)
            # La fenêtre de lissage ne voit qu'un côté au bord d'une longueur :
            # vitesse ~4× plus bruitée sur la demi-fenêtre.
            index = np.flatnonzero(length)
            half = window // 2
            confidence[np.r_[index[:half], index[-half:]]] *= EDGE_CONFIDENCE_FACTOR

    return Trajectory(
        t=t,
        x=x_smooth,
        d=np.where(outbound, x_smooth, 2 * POOL_LENGTH - x_smooth),
        v=np.where(outbound, dx_dt, -dx_dt),
        confidence=confidence,
    )


def _frame_period(t: FloatArray) -> float:
    if t.size < 2:
        raise ValueError("au moins deux frames attendues")
    steps = np.diff(t)
    dt = float(np.median(steps))
    if dt <= 0 or not np.allclose(steps, dt, rtol=1e-3):
        raise ValueError("les instants doivent être à pas constant (une valeur par frame)")
    return dt


def _odd_window(duration_s: float, dt: float) -> int:
    """Nombre impair d'échantillons couvrant `duration_s`, au moins 3."""
    n = max(3, round(duration_s / dt))
    return n if n % 2 else n + 1


def _outliers(x: FloatArray, valid: BoolArray, window: int) -> BoolArray:
    """Détections trop loin de la médiane glissante (saut vers un autre nageur, reflet…)."""
    if np.count_nonzero(valid) < 2:
        return np.zeros_like(valid)
    index = np.arange(x.size)
    filled = np.interp(index, index[valid], x[valid])
    median = median_filter(filled, size=window, mode="nearest")
    return valid & (np.abs(filled - median) > OUTLIER_THRESHOLD_M)


def _fill_gaps(
    t: FloatArray, x: FloatArray, confidence: FloatArray, valid: BoolArray, dt: float
) -> tuple[FloatArray, FloatArray]:
    """Interpole linéairement les trous et abaisse la confiance des points interpolés."""
    x_filled = np.interp(t, t[valid], x[valid])
    filled_confidence = np.where(valid, confidence, 0.0)

    change = np.diff((~valid).astype(np.int8))
    starts = np.flatnonzero(change == 1) + 1
    ends = np.flatnonzero(change == -1) + 1  # exclusif ; les bords sont valides
    for start, end in zip(starts, ends, strict=True):
        before, after = start - 1, end
        if t[after] - t[before] <= MAX_INTERPOLATED_GAP_S + dt / 2:
            neighbours = min(filled_confidence[before], filled_confidence[after])
            filled_confidence[start:end] = INTERPOLATED_CONFIDENCE_FACTOR * neighbours
    return x_filled, filled_confidence


def _smooth(x: FloatArray, window: int, dt: float) -> tuple[FloatArray, FloatArray]:
    """Position lissée et sa dérivée ; fenêtre réduite si la longueur est courte."""
    window = min(window, x.size if x.size % 2 else x.size - 1)
    if window <= SMOOTHING_POLYORDER:
        speed = np.gradient(x, dt) if x.size > 1 else np.zeros_like(x)
        return x.copy(), speed
    smoothed = savgol_filter(x, window, SMOOTHING_POLYORDER, mode="interp")
    speed = savgol_filter(x, window, SMOOTHING_POLYORDER, deriv=1, delta=dt, mode="interp")
    return smoothed, speed
