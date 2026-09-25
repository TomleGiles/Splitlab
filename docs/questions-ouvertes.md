# Questions ouvertes — choix d'implémentation à valider

Choix faits faute de définition dans `CLAUDE.md`. Une fois tranchés, reporter la
décision dans `CLAUDE.md` et retirer l'entrée.

Les questions 1 à 5 (temps de réaction, passages aux murs, coulée après virage,
marqueurs de cycle, saut de `d` au virage) ont été tranchées le 2026-09-25 et
reportées dans `CLAUDE.md`.

## Détection (`cv/detection.py`)

6. **Avant du nageur ou tête ?** La détection sans apprentissage donne l'étendue de
   la zone occupée par le nageur ; le point suivi est son avant (bras compris). Les
   passages usuels se lisent au passage de la tête, qui est en retrait d'une
   distance variable selon la phase de bras (jusqu'à ~0,7 m). Après lissage, il
   reste probablement un décalage moyen de quelques dizaines de centimètres, soit
   ~0,1–0,2 s sur un passage : au-delà de la cible ±0,05 s. **À mesurer** sur des
   vidéos réelles annotées (`cv.eval`) avant de corriger — ne pas deviner la valeur.
