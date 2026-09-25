// Mise en forme d'affichage (français). Les valeurs reçues sont en unités SI.

const formatters = new Map<number, Intl.NumberFormat>();

function formatter(digits: number): Intl.NumberFormat {
  let f = formatters.get(digits);
  if (!f) {
    f = new Intl.NumberFormat("fr-FR", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
    formatters.set(digits, f);
  }
  return f;
}

export function num(value: number | null, digits = 2): string {
  return value === null ? "—" : formatter(digits).format(value);
}

export function pct(value: number | null): string {
  if (value === null) return "—";
  const sign = value > 0 ? "+" : value < 0 ? "−" : "";
  return `${sign}${formatter(1).format(Math.abs(value))} %`;
}
