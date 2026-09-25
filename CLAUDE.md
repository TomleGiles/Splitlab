# CLAUDE.md — Splitlab

## Le produit

SaaS d'**analyse vidéo automatisée de course de natation** pour structures de haut niveau (pôles espoirs, clubs de niveau national, fédérations).
Aujourd'hui un analyste passe 20 à 40 min par course à extraire à la main passages, coulées, cadence et virages depuis une vidéo filmée au bord du bassin. Splitlab produit la même fiche en ~2 min.

**Utilisateur** : l'entraîneur ou l'analyste vidéo du club, pas le nageur.
**Livrable clé** : une fiche de course (web + PDF) lisible par un coach, avec comparaison à des références.

## Périmètre MVP — ne pas en sortir

- **Nage** : crawl uniquement. Aucune autre nage tant que le crawl n'est pas validé sur vidéos réelles.
- **Épreuve** : 50 NL en **bassin de 25 m** (un virage, pour pouvoir mesurer le virage).
- **Captation** : une seule caméra fixe, vue latérale depuis le bord, à hauteur des gradins, **immobile pendant toute la course** (trépied, pas de zoom), les deux murs visibles. 25 ou 50 fps.
- **Un nageur analysé par vidéo** (sa ligne est désignée par l'utilisateur).
- **Flux** : upload vidéo → calibration du bassin → traitement asynchrone → correction semi-manuelle → fiche de course + export PDF.

Hors MVP (refuser ou noter dans `docs/backlog.md`, ne pas implémenter) : autres nages, multi-caméra, caméra sous-marine, comptage d'ondulations, angle du corps, analyse en direct, app mobile native, multi-nageurs simultanés, entraînement de modèles custom.

## Ordre de construction

1. **Pipeline en ligne de commande**, validé sur 5–10 courses réelles contre la vérité terrain (cibles de la section « Évaluation »). Tant que ces cibles ne sont pas tenues, le front se limite à la fiche de course d'une **course simulée** (`api/demo.py`, `GET /api/demo/race`), toujours affichée comme démonstration.
2. **Web V0** : la stack simplifiée ci-dessous, pour quelques clubs pilotes.
3. **Passage à l'échelle** (colonne « Plus tard ») seulement quand un besoin concret le justifie (charge, nombre de clubs, hébergement). Ne pas l'anticiper sans décision explicite.

Non négociable **dès le V0** : traitement vidéo hors de la requête HTTP, multi-tenant strict par `club_id`, suppression réelle des vidéos (voir « Données et RGPD »).

## Stack

| Couche | V0 | Plus tard |
|---|---|---|
| Pipeline CV | Python 3.12, OpenCV (homographie, redressement du couloir, soustraction de fond), NumPy/SciPy — **sans apprentissage**. YOLO pré-entraîné écarté : il ne détecte pas les nageurs en nage (`docs/essais/2026-09-25-detection-yolo.md`) | Détecteur « nageur » entraîné, si la soustraction de fond ne suffit pas |
| API | FastAPI, Pydantic v2, SQLAlchemy 2 + Alembic | — |
| Jobs | Un process worker qui dépile une table `jobs` dans PostgreSQL (`SELECT … FOR UPDATE SKIP LOCKED`) | Redis + arq |
| Base | PostgreSQL 16 | — |
| Stockage vidéo | Disque local chiffré, derrière une interface `VideoStorage` | S3-compatible |
| Front | React + Vite (SPA servie par FastAPI), TypeScript strict, Tailwind, Recharts | Next.js si besoin de SSR |
| PDF | Impression navigateur de la page fiche (CSS `@media print`) | WeasyPrint, même template |
| Outillage | `uv` (Python), `pnpm` (front), Docker Compose (Postgres seul), pytest, Ruff, mypy, ESLint | — |

## Arborescence

```
.
├── cv/                  # pipeline de vision, librairie Python pure, sans dépendance web
│   ├── domain.py        # types partagés : Event, Trajectory, MetricValue
│   ├── calibration.py   # homographie image → bassin
│   ├── detection.py     # nageur dans son couloir : bande redressée, soustraction de fond
│   ├── trajectory.py    # lissage, position x(t) en mètres, vitesse v(t)
│   ├── events.py        # détection départ, coulée, reprise de nage, cycles, virage, arrivée
│   ├── metrics.py       # calcul des métriques à partir des événements (fonctions pures)
│   └── pipeline.py      # orchestration vidéo → RaceAnalysis
├── api/                 # FastAPI, modèles DB, worker de jobs, stockage vidéo
├── web/                 # front React + Vite (build servi par FastAPI)
├── benchmarks/          # valeurs de référence élite (JSON versionné, sources citées)
├── tests/
│   ├── cv/              # tests unitaires sur trajectoires synthétiques
│   └── fixtures/        # petits extraits vidéo + vérité terrain annotée
├── data/                # vidéos réelles — JAMAIS commitées (dans .gitignore)
└── docs/
```

`cv/` ne doit jamais importer `api/`. Le pipeline doit être exécutable seul :
```
uv run python -m cv.pipeline --video path.mp4 --calib calib.json --lane 4 --out result.json
```

## Commandes

```
docker compose up -d            # Postgres
uv run pytest                   # tests Python
uv run ruff check . && uv run mypy cv api
uv run alembic upgrade head
uv run python -m api.worker     # worker de traitement vidéo
uv run uvicorn api.main:app --reload   # sert aussi web/dist → http://localhost:8000
cd web && pnpm build            # build du front servi par FastAPI (V0)
cd web && pnpm dev              # dev front sur :5173, /api proxifié vers :8000
```

Avant de considérer une tâche terminée : ruff, mypy et pytest passent.

## Modèle du domaine

### Conventions (non négociables)
- **Unités SI partout en interne** : secondes (float), mètres, m/s. Conversion d'affichage uniquement dans le front.
- **Temps** : `t = 0` au signal de départ, pas au début de la vidéo. Stocker aussi l'index de frame pour chaque événement.
- **Espace** : repère bassin. `x` = distance au mur de départ (0 → 25 m), `y` = position transversale. Pour le retour après virage, `x` redescend de 25 à 0 ; la **distance parcourue** `d` (0 → 50 m) est une grandeur séparée.
- Toute métrique porte un **score de confiance** (0–1) et un flag `manually_corrected`.

### Calibration
L'utilisateur clique au moins 4 repères sur une frame (coins de la ligne, marques 5 m / 15 m des lignes d'eau) et saisit leurs coordonnées réelles. On calcule l'homographie avec `cv2.findHomography` (RANSAC). Afficher l'erreur de reprojection ; refuser au-delà d'un seuil (> 0,3 m).

### Événements détectés
`start_signal`, `block_off` (dernier contact avec le plot), `entry` (entrée dans l'eau), `breakout` (reprise de nage), `stroke_cycle[]` (un cycle = deux bras en crawl), `wall_in` / `wall_out` (virage : contact des pieds / poussée), `finish` (touche).

- Un `stroke_cycle` marque le **début** d'un cycle : l'entrée dans l'eau de la main côté caméra. n marqueurs délimitent n − 1 cycles.
- Le point suivi est **l'avant du nageur détecté** (front de la zone qui diffère de l'eau : bras et tête). Il fait demi-tour avant le mur : `d` saute d'autant au virage. Les passages et le temps de virage, lus par première traversée, n'en dépendent pas ; les points du profil de vitesse dans ce saut héritent de la confiance réduite des bords de longueur.

Le signal de départ : détection du bip audio si disponible, sinon saisie manuelle de la frame.

### Métriques (définitions de référence)
| Métrique | Définition |
|---|---|
| Temps de réaction | `start_signal` → `block_off`. Si non mesurable : `null`, jamais estimé |
| Passages | temps à 15 m, 25 m, 35 m, 50 m (distance parcourue). 15 et 35 m : première traversée par le point suivi ; 25 m : `wall_in` ; 50 m : `finish` (le point suivi n'atteint jamais le mur) |
| Distance de coulée | distance au mur quitté au `breakout` : `x` après le départ, `25 − x` après le virage |
| Fréquence (SR) | cycles/min, par section de nage libre (aller : cycles avant `wall_in`, retour : après `wall_out`) |
| Amplitude (SL) | m/cycle = `v_moyenne / (SR / 60)`, `v_moyenne` prise du premier au dernier marqueur de la section |
| Indice de nage (SI) | `v × SL` |
| Temps de virage | de 5 m avant le mur à 5 m après (d = 20 → 30 m) |
| Vitesse d'arrivée | vitesse moyenne sur les 5 derniers mètres |
| Profil de vitesse | `v(d)` lissée, échantillonnée tous les 0,5 m |

Toutes ces fonctions vivent dans `cv/metrics.py`, sont **pures** (pas d'I/O, pas d'OpenCV) et testées sur trajectoires synthétiques dont le résultat est connu analytiquement.

### Ce qui se vend (priorité dans la fiche)
1. Profil de vitesse continu sur la course.
2. Lecture croisée fréquence / amplitude par section (diagnostic de fatigue : la fréquence tient mais l'amplitude chute, ou l'inverse).
3. Comparaison aux références élite (`benchmarks/`) exprimée en temps gagné/perdu : « −0,3 s au virage ».
4. Suivi longitudinal d'un nageur sur la saison.

Les chiffres bruts seuls ne suffisent pas : chaque métrique affichée doit avoir une comparaison ou une interprétation.

Lecture croisée SR / SL (retour vs aller) : une variation est « stable » si elle reste dans ±3 %. Fréquence stable + amplitude en baisse = perte d'efficacité par cycle (fatigue) ; amplitude stable + fréquence en baisse = baisse de rythme ; fréquence en hausse + amplitude en baisse = compensation. Seuil à ajuster avec les coachs pilotes.

## Fiabilité et mode semi-manuel

- Fiables avec bon tracking + bonne homographie : position, vitesse, passages, coulées, temps de virage.
- **Fragiles** (éclaboussures, remous) : cycles de bras → fréquence et amplitude. Le mode semi-manuel est prévu **dès le départ**, pas en rattrapage.
- Le front propose une timeline vidéo où l'utilisateur ajoute / déplace / supprime des événements ; les métriques sont recalculées côté serveur à partir des événements corrigés (jamais éditées directement).
- Si la confiance d'un événement est faible, l'afficher comme « à vérifier » plutôt que de masquer l'incertitude. Seuil : confiance < 0,7.

## Évaluation du pipeline

- `tests/fixtures/` contient des extraits annotés à la main (vérité terrain des événements en JSON).
- `uv run python -m cv.eval` compare le pipeline à la vérité terrain et sort l'erreur par métrique (MAE en s ou en m).
- Toute modification de `cv/` qui dégrade l'erreur d'une métrique doit être signalée explicitement.
- Cibles MVP : passages ±0,05 s, distance de coulée ±0,5 m, temps de virage ±0,1 s.

## Données et RGPD

- Les nageurs des pôles espoirs sont souvent **mineurs** : vidéos = données personnelles sensibles.
- Stockage chiffré, accès limité au club propriétaire (multi-tenant strict : toute requête filtrée par `club_id`).
- Durée de conservation configurable, suppression réelle des vidéos (pas de soft delete sur les fichiers).
- Aucune vidéo réelle dans le repo, les logs ou les tests. Pas de nom de nageur dans les logs.
- Les benchmarks sont issus de vidéos publiques de championnats ; conserver la source de chaque valeur.

## Consignes de travail pour Claude

- Travailler par petites étapes testables ; commencer par `cv/metrics.py` + tests, puis `trajectory`, puis `events`, puis la détection.
- Ne pas entraîner ni fine-tuner de modèle sans demande explicite ; la détection est sans apprentissage (décision du 2026-09-25).
- Ne pas ajouter de dépendance lourde sans justification courte.
- Rester sur la stack V0 : pas de Redis, S3, Next.js ni WeasyPrint sans décision explicite.
- Les choix d'interprétation en attente de validation sont listés dans `docs/questions-ouvertes.md`.
- En cas de doute sur une définition métier (natation), demander plutôt qu'inventer ; les définitions de ce fichier font foi.
- Mettre à jour ce fichier quand une décision d'architecture ou une définition change.
