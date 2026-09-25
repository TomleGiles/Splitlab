"""Fiche de course : métriques + interprétations, prêtes pour le front.

Unités SI ; la mise en forme d'affichage reste dans le front.
"""

import math
from typing import Literal

from pydantic import BaseModel

from cv.domain import MetricValue
from cv.metrics import SPLIT_DISTANCES, RaceMetrics, SectionMetrics

REVIEW_THRESHOLD = 0.7  # confiance en dessous : « à vérifier »
STABLE_CHANGE_PCT = 3.0  # variation retour/aller considérée comme stable

Trend = Literal["stable", "up", "down"]

_DIAGNOSES: dict[tuple[Trend, Trend], str] = {
    ("stable", "down"): (
        "Fréquence maintenue mais amplitude en baisse : "
        "perte d'efficacité par cycle, signe de fatigue."
    ),
    ("down", "stable"): "Amplitude maintenue mais fréquence en baisse : baisse de rythme.",
    ("up", "down"): "Fréquence en hausse pour compenser une amplitude en baisse.",
    ("down", "down"): "Fréquence et amplitude en baisse : baisse nette sur le retour.",
    ("stable", "stable"): "Fréquence et amplitude stables entre l'aller et le retour.",
    ("down", "up"): "Fréquence en baisse, amplitude en hausse : nage plus allongée.",
    ("up", "up"): "Fréquence et amplitude en hausse sur le retour.",
    ("up", "stable"): "Fréquence en hausse, amplitude maintenue : accélération sur le retour.",
    ("stable", "up"): "Amplitude en hausse, fréquence maintenue : meilleure efficacité.",
}


class MetricOut(BaseModel):
    value: float | None
    confidence: float
    manually_corrected: bool
    to_check: bool


class SplitOut(BaseModel):
    distance: float  # m
    time: MetricOut  # s
    segment_time: float | None  # s depuis le passage précédent
    segment_speed: float | None  # m/s sur le segment


class SectionOut(BaseModel):
    label: str
    stroke_rate: MetricOut  # cycles/min
    stroke_length: MetricOut  # m/cycle
    stroke_index: MetricOut  # m²/s


class StrokeReadingOut(BaseModel):
    stroke_rate_change_pct: float | None
    stroke_length_change_pct: float | None
    stroke_index_change_pct: float | None
    diagnosis: str | None


class ProfilePointOut(BaseModel):
    d: float  # m
    v: float | None  # m/s
    to_check: bool


class RaceSheet(BaseModel):
    is_demo: bool
    title: str
    reaction_time: MetricOut
    splits: list[SplitOut]
    underwater_start: MetricOut
    underwater_turn: MetricOut
    sections: list[SectionOut]
    stroke_reading: StrokeReadingOut
    turn_time: MetricOut
    finish_speed: MetricOut
    velocity_profile: list[ProfilePointOut]
    benchmarks_available: bool


def metric_out(metric: MetricValue) -> MetricOut:
    return MetricOut(
        value=metric.value,
        confidence=metric.confidence,
        manually_corrected=metric.manually_corrected,
        to_check=metric.value is not None and metric.confidence < REVIEW_THRESHOLD,
    )


def splits_out(splits: dict[float, MetricValue]) -> list[SplitOut]:
    result = []
    previous_distance = 0.0
    previous_time: float | None = 0.0
    for distance in SPLIT_DISTANCES:
        time = splits[distance].value
        segment = None if time is None or previous_time is None else time - previous_time
        result.append(
            SplitOut(
                distance=distance,
                time=metric_out(splits[distance]),
                segment_time=segment,
                segment_speed=None if segment is None else (distance - previous_distance) / segment,
            )
        )
        previous_distance, previous_time = distance, time
    return result


def _change_pct(before: MetricValue, after: MetricValue) -> float | None:
    if before.value is None or after.value is None or before.value == 0:
        return None
    return 100.0 * (after.value - before.value) / before.value


def _trend(change_pct: float) -> Trend:
    if abs(change_pct) <= STABLE_CHANGE_PCT:
        return "stable"
    return "up" if change_pct > 0 else "down"


def stroke_reading(outbound: SectionMetrics, inbound: SectionMetrics) -> StrokeReadingOut:
    """Lecture croisée fréquence / amplitude : retour comparé à l'aller."""
    sr_change = _change_pct(outbound.stroke_rate, inbound.stroke_rate)
    sl_change = _change_pct(outbound.stroke_length, inbound.stroke_length)
    diagnosis = None
    if sr_change is not None and sl_change is not None:
        diagnosis = _DIAGNOSES[_trend(sr_change), _trend(sl_change)]
    return StrokeReadingOut(
        stroke_rate_change_pct=sr_change,
        stroke_length_change_pct=sl_change,
        stroke_index_change_pct=_change_pct(outbound.stroke_index, inbound.stroke_index),
        diagnosis=diagnosis,
    )


def _section_out(label: str, section: SectionMetrics) -> SectionOut:
    return SectionOut(
        label=label,
        stroke_rate=metric_out(section.stroke_rate),
        stroke_length=metric_out(section.stroke_length),
        stroke_index=metric_out(section.stroke_index),
    )


def build_sheet(metrics: RaceMetrics, title: str, is_demo: bool) -> RaceSheet:
    outbound, inbound = metrics.sections
    profile = metrics.velocity_profile
    return RaceSheet(
        is_demo=is_demo,
        title=title,
        reaction_time=metric_out(metrics.reaction_time),
        splits=splits_out(metrics.splits),
        underwater_start=metric_out(metrics.underwater_start),
        underwater_turn=metric_out(metrics.underwater_turn),
        sections=[_section_out("Aller", outbound), _section_out("Retour", inbound)],
        stroke_reading=stroke_reading(outbound, inbound),
        turn_time=metric_out(metrics.turn_time),
        finish_speed=metric_out(metrics.finish_speed),
        velocity_profile=[
            ProfilePointOut(
                d=float(d),
                v=None if math.isnan(v) else float(v),
                to_check=not math.isnan(v) and c < REVIEW_THRESHOLD,
            )
            for d, v, c in zip(profile.d, profile.v, profile.confidence, strict=True)
        ],
        # Aucune valeur élite sourcée dans benchmarks/ pour l'instant.
        benchmarks_available=False,
    )
