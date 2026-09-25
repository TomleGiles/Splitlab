import {
  Area,
  CartesianGrid,
  ComposedChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from "recharts";

import { num } from "./format";
import type { ProfilePoint } from "./types";
import { ToCheck } from "./ui";

const DISTANCE_TICKS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
const AXIS_TICK = { fill: "var(--muted)", fontSize: 12 };

/** Plages contiguës de points « à vérifier », élargies d'un demi-pas. */
function toCheckRanges(points: ProfilePoint[]): Array<[number, number]> {
  const step = points.length > 1 ? points[1]!.d - points[0]!.d : 0.5;
  const ranges: Array<[number, number]> = [];
  for (const p of points) {
    if (!p.to_check) continue;
    const last = ranges.at(-1);
    if (last && p.d - last[1] <= step + 1e-9) last[1] = p.d;
    else ranges.push([p.d, p.d]);
  }
  return ranges.map(([a, b]) => [Math.max(0, a - step / 2), Math.min(50, b + step / 2)]);
}

function ProfileTooltip({ active, payload }: TooltipContentProps) {
  const point = payload?.[0]?.payload as ProfilePoint | undefined;
  if (!active || !point || point.v === null) return null;
  return (
    <div className="rounded-md border border-line bg-surface px-3 py-2 text-sm shadow-sm">
      <div className="text-ink-2">{num(point.d, 1)} m</div>
      <div className="font-semibold">{num(point.v)} m/s</div>
      {point.to_check && <ToCheck />}
    </div>
  );
}

export function VelocityChart({ points }: { points: ProfilePoint[] }) {
  // Graduations rondes : 0, 1, 2… m/s jusqu'au pic arrondi au-dessus.
  const top = Math.max(1, Math.ceil(Math.max(...points.map((p) => p.v ?? 0))));
  const speedTicks = Array.from({ length: top + 1 }, (_, i) => i);
  return (
    <div>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={points} margin={{ top: 8, right: 12, bottom: 0, left: -8 }}>
            <CartesianGrid vertical={false} stroke="var(--grid)" />
            {toCheckRanges(points).map(([x1, x2]) => (
              <ReferenceArea key={x1} x1={x1} x2={x2} fill="var(--warning-wash)" stroke="none" />
            ))}
            <ReferenceLine
              x={25}
              stroke="var(--axis)"
              label={{ value: "Virage", position: "insideTopLeft", ...AXIS_TICK }}
            />
            <XAxis
              dataKey="d"
              type="number"
              domain={[0, 50]}
              ticks={DISTANCE_TICKS}
              tickFormatter={(d: number) => `${d} m`}
              tick={AXIS_TICK}
              tickLine={false}
              stroke="var(--axis)"
            />
            <YAxis
              domain={[0, top]}
              ticks={speedTicks}
              tickFormatter={(v: number) => num(v, 0)}
              tick={AXIS_TICK}
              tickLine={false}
              axisLine={false}
              width={44}
            />
            <Tooltip
              content={ProfileTooltip}
              cursor={{ stroke: "var(--axis)", strokeWidth: 1 }}
              isAnimationActive={false}
            />
            <Area
              dataKey="v"
              type="monotone"
              stroke="var(--series-1)"
              strokeWidth={2}
              fill="var(--series-1)"
              fillOpacity={0.1}
              connectNulls={false}
              dot={false}
              activeDot={{ r: 4, fill: "var(--series-1)", stroke: "var(--surface)", strokeWidth: 2 }}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-2">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-3 w-5 rounded-sm bg-[var(--warning-wash)]" />
          Zone à vérifier (confiance &lt; 0,7)
        </span>
        <span>Axe vertical : vitesse en m/s</span>
      </div>
    </div>
  );
}

export function ProfileTable({ points }: { points: ProfilePoint[] }) {
  const rows = points.filter((p) => Math.abs(p.d / 2.5 - Math.round(p.d / 2.5)) < 1e-9);
  return (
    <details className="no-print mt-4 text-sm">
      <summary className="cursor-pointer text-ink-2">Voir les valeurs (tous les 2,5 m)</summary>
      <div className="mt-2 grid grid-cols-3 gap-x-6 gap-y-1 tabular-nums sm:grid-cols-7">
        {rows.map((p) => (
          <div key={p.d} className="flex justify-between gap-2">
            <span className="text-muted">{num(p.d, 1)} m</span>
            <span>
              {num(p.v)}
              {p.to_check && <span title="à vérifier"> ⚠</span>}
            </span>
          </div>
        ))}
      </div>
    </details>
  );
}
