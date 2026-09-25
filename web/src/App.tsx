import { useEffect, useState } from "react";

import { num, pct } from "./format";
import type { Metric, RaceSheet } from "./types";
import { Card, MetricValue, Stat, ToCheck } from "./ui";
import { ProfileTable, VelocityChart } from "./VelocityChart";

export function App() {
  const [sheet, setSheet] = useState<RaceSheet | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/demo/race")
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<RaceSheet>;
      })
      .then(setSheet)
      .catch((e: unknown) => setError(String(e)));
  }, []);

  return (
    <div className="mx-auto max-w-5xl px-4 py-6 sm:py-10">
      <header className="flex items-baseline justify-between border-b border-line pb-4">
        <span className="text-lg font-semibold tracking-tight">Splitlab</span>
        <span className="text-sm text-muted">Fiche de course</span>
      </header>
      {error && (
        <p className="mt-8 text-ink-2">
          Impossible de charger la fiche ({error}). L'API tourne-t-elle sur le port 8000 ?
        </p>
      )}
      {!sheet && !error && <p className="mt-8 text-muted">Chargement…</p>}
      {sheet && <Sheet sheet={sheet} />}
    </div>
  );
}

function Sheet({ sheet }: { sheet: RaceSheet }) {
  const final = sheet.splits.at(-1)?.time;
  return (
    <main className="mt-6 space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-semibold tracking-tight">{sheet.title}</h1>
        {sheet.is_demo && (
          <span className="rounded-full border border-line bg-surface px-3 py-1 text-xs text-ink-2">
            Démonstration · course simulée, aucune vidéo réelle
          </span>
        )}
      </div>

      <div className="grid gap-4 md:grid-cols-[minmax(12rem,auto)_1fr]">
        <div className="flex flex-col justify-center rounded-xl border border-line bg-surface p-5">
          <span className="text-sm text-ink-2">Temps final</span>
          <span className="mt-1 text-5xl font-semibold tracking-tight">
            {final ? <MetricValue metric={final} unit="s" /> : "—"}
          </span>
          {final?.to_check && <ToCheck />}
        </div>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
          <Stat label="Réaction" metric={sheet.reaction_time} unit="s" caption="Signal → dernier contact plot" />
          <Stat label="Virage" metric={sheet.turn_time} unit="s" caption="De 20 m à 30 m" />
          <Stat label="Vitesse d'arrivée" metric={sheet.finish_speed} unit="m/s" caption="5 derniers mètres" />
          <Stat label="Coulée départ" metric={sheet.underwater_start} unit="m" digits={1} caption="Depuis le mur de départ" />
          <Stat label="Coulée virage" metric={sheet.underwater_turn} unit="m" digits={1} caption="Depuis le mur de virage" />
        </div>
      </div>

      <Card title="Profil de vitesse" subtitle="Vitesse selon la distance parcourue, lissée">
        <VelocityChart points={sheet.velocity_profile} />
        <ProfileTable points={sheet.velocity_profile} />
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <StrokeCard sheet={sheet} />
        <SplitsCard sheet={sheet} />
      </div>

      {!sheet.benchmarks_available && (
        <p className="rounded-xl border border-dashed border-line px-5 py-4 text-sm text-ink-2">
          Comparaison aux références élite : pas encore disponible. Les valeurs de référence
          seront ajoutées avec leur source (vidéos publiques de championnats).
        </p>
      )}

      <footer className="border-t border-line pt-4 text-xs text-muted">
        Métriques recalculées à partir des événements détectés ou corrigés. Une valeur dont la
        confiance est inférieure à 0,7 est marquée « à vérifier ».
      </footer>
    </main>
  );
}

function Cell({ metric, unit, digits = 2 }: { metric: Metric; unit: string; digits?: number }) {
  return (
    <td className="py-2 text-right tabular-nums">
      <div>
        <MetricValue metric={metric} unit={unit} digits={digits} />
      </div>
      {metric.to_check && <ToCheck />}
    </td>
  );
}

function StrokeCard({ sheet }: { sheet: RaceSheet }) {
  const [outbound, inbound] = sheet.sections;
  const reading = sheet.stroke_reading;
  if (!outbound || !inbound) return null;
  const rows = [
    { label: "Fréquence", key: "stroke_rate", unit: "c/min", digits: 1, change: reading.stroke_rate_change_pct },
    { label: "Amplitude", key: "stroke_length", unit: "m/cycle", digits: 2, change: reading.stroke_length_change_pct },
    { label: "Indice de nage", key: "stroke_index", unit: "m²/s", digits: 2, change: reading.stroke_index_change_pct },
  ] as const;
  return (
    <Card title="Fréquence et amplitude" subtitle="Retour comparé à l'aller">
      {reading.diagnosis && (
        <p className="mb-3 rounded-lg bg-plane px-3 py-2 text-sm">{reading.diagnosis}</p>
      )}
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs text-muted">
            <th className="py-2 font-normal" />
            <th className="py-2 text-right font-normal">{outbound.label}</th>
            <th className="py-2 text-right font-normal">{inbound.label}</th>
            <th className="py-2 text-right font-normal">Variation</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key} className="border-b border-line last:border-0 align-top">
              <td className="py-2 text-ink-2">{row.label}</td>
              <Cell metric={outbound[row.key]} unit={row.unit} digits={row.digits} />
              <Cell metric={inbound[row.key]} unit={row.unit} digits={row.digits} />
              <td className="py-2 text-right tabular-nums">{pct(row.change)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-3 text-xs text-muted">
        Variation « stable » dans ±3 %. Cycle = deux bras.
      </p>
    </Card>
  );
}

function SplitsCard({ sheet }: { sheet: RaceSheet }) {
  let previous = 0;
  return (
    <Card title="Passages" subtitle="Temps cumulés et vitesse moyenne par segment">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs text-muted">
            <th className="py-2 font-normal">Distance</th>
            <th className="py-2 text-right font-normal">Passage</th>
            <th className="py-2 text-right font-normal">Segment</th>
            <th className="py-2 text-right font-normal">Vitesse</th>
          </tr>
        </thead>
        <tbody>
          {sheet.splits.map((split) => {
            const from = previous;
            previous = split.distance;
            return (
              <tr key={split.distance} className="border-b border-line last:border-0 align-top">
                <td className="py-2 text-ink-2">{num(split.distance, 0)} m</td>
                <Cell metric={split.time} unit="s" />
                <td className="py-2 text-right tabular-nums">
                  {num(split.segment_time)} s
                  <div className="text-xs text-muted">
                    {num(from, 0)}–{num(split.distance, 0)} m
                  </div>
                </td>
                <td className="py-2 text-right tabular-nums">{num(split.segment_speed)} m/s</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-3 text-xs text-muted">25 m : contact au mur · 50 m : touche d'arrivée.</p>
    </Card>
  );
}
