"""Application FastAPI. En V0, sert aussi le build du front (`web/dist`)."""

from functools import cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.demo import simulate_demo_race
from api.sheet import RaceSheet, build_sheet
from cv.metrics import compute_race_metrics

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"

app = FastAPI(title="Splitlab")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@cache
def _demo_sheet() -> RaceSheet:
    trajectory, events = simulate_demo_race()
    metrics = compute_race_metrics(trajectory, events)
    return build_sheet(metrics, title="50 NL — bassin de 25 m", is_demo=True)


@app.get("/api/demo/race")
def demo_race() -> RaceSheet:
    """Fiche d'une course simulée (données synthétiques) — pas de vidéo réelle."""
    return _demo_sheet()


if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
