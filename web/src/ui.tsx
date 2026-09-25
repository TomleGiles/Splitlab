import type { ReactNode } from "react";

import { num } from "./format";
import type { Metric } from "./types";

export function Card({
  title,
  subtitle,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-xl border border-line bg-surface p-5 ${className}`}>
      <h2 className="text-base font-semibold">{title}</h2>
      {subtitle && <p className="mt-0.5 text-sm text-ink-2">{subtitle}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}

/** Statut « à vérifier » : icône + libellé, jamais la couleur seule. */
export function ToCheck() {
  return (
    <span className="inline-flex items-center gap-1 text-xs font-medium text-ink-2">
      <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
        <path d="M6 1 11.2 10.5H.8Z" fill="var(--warning)" />
        <path d="M6 4.4v3M6 8.6v.1" stroke="#0b0b0b" strokeWidth="1.3" strokeLinecap="round" />
      </svg>
      à vérifier
    </span>
  );
}

export function MetricValue({
  metric,
  unit,
  digits = 2,
}: {
  metric: Metric;
  unit: string;
  digits?: number;
}) {
  if (metric.value === null) {
    return <span className="text-muted">non mesurable</span>;
  }
  return (
    <span>
      {num(metric.value, digits)}
      <span className="ml-0.5 text-ink-2"> {unit}</span>
    </span>
  );
}

export function Stat({
  label,
  metric,
  unit,
  digits = 2,
  caption,
}: {
  label: string;
  metric: Metric;
  unit: string;
  digits?: number;
  caption: string;
}) {
  return (
    <div className="rounded-lg border border-line bg-surface p-4">
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm text-ink-2">{label}</span>
        {metric.to_check && <ToCheck />}
      </div>
      <div className="mt-1 text-2xl font-semibold">
        <MetricValue metric={metric} unit={unit} digits={digits} />
      </div>
      <p className="mt-1 text-xs text-muted">{caption}</p>
    </div>
  );
}
