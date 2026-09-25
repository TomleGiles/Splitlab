# Questions ouvertes — choix d'implémentation à valider

Choix faits faute de définition dans `CLAUDE.md`. Une fois tranchés, reporter la
décision dans `CLAUDE.md` et retirer l'entrée.

## Métriques (`cv/metrics.py`)

1. **Temps de réaction** — aucun événement ne marque le dernier contact avec le plot.
   Actuellement toujours `null`. Proposition : ajouter un événement `block_off`.
2. **Passages à 25 m et 50 m** — le point suivi n'atteint jamais le mur ; on prend
   le temps de `wall_in` (25 m) et de `finish` (50 m).
3. **Coulée après virage** — mesurée depuis le mur de virage : `25 − x` au `breakout`
   (et non `x` brut).
4. **Cycles de bras** — un `stroke_cycle` marque le début d'un cycle ; n marqueurs
   = n − 1 cycles. SL = distance entre premier et dernier marqueur / nombre de cycles.

## Trajectoire (`cv/trajectory.py`)

5. **Discontinuité de `d` au virage** — le point suivi fait demi-tour avant le mur
   (tête à ~1–1,5 m). `d = x` à l'aller et `d = 50 − x` au retour, donc `d` saute
   d'environ 2–3 m à l'instant du demi-tour. Les passages et le temps de virage
   (lus par première traversée) n'en sont pas affectés ; dans ce saut, le profil de
   vitesse est interpolé entre la vitesse avant et après le demi-tour.
