# Essai — détection des nageurs avec YOLO pré-entraîné (2026-09-25)

## Vidéo

Course publique : 50 NL messieurs, China Open 2026 (finale A). Filmée au téléphone
depuis les gradins, 1920×1080 à 60 fps, 28,6 s. Stockée dans `data/` (jamais commitée).

Hors conditions MVP : **bassin de 50 m** (pas de virage) et **caméra à la main qui
panoramique** pour suivre la course → pas de calibration fixe possible, donc pas de
fiche de course fiable à partir de cette vidéo.

## Protocole

Détection de la classe COCO `person` sur 6 images (0,5 s → 22 s), seuil de confiance
bas (0,15 puis 0,05), avec `yolo11n` (imgsz 1280) puis `yolo11x` (imgsz 1920).

## Résultats

| Moment | Détecté | Nageurs dans l'eau détectés |
|---|---|---|
| Sur les plots / plongeon (t = 0,5 s, 4 s) | officiels, public, nageurs sur les plots (conf. 0,4–0,65) | — |
| En nage (t = 8, 12, 17 s), `yolo11n` | officiels, public | **0 / 8** |
| En nage (t = 8, 12, 17 s), `yolo11x` | officiels, public | **≤ 1 / 8, conf. ≤ 0,06** |

## Conclusion

Le détecteur `person` pré-entraîné **ne voit pas un nageur en nage** (corps immergé,
éclaboussures, vue de loin et de biais). Le choix de la stack « YOLO pré-entraîné,
détection `person` » ne tient pas pour la phase de nage ; il reste utilisable pour
le départ (nageurs sur les plots).

Pistes (décision à prendre, voir CLAUDE.md « ne pas entraîner sans demande ») :
1. Détection par couloir sans apprentissage : caméra fixe + calibration → bande du
   couloir connue ; nageur = zone qui diffère de l'eau de fond (soustraction de fond,
   écume, tête/bonnet), front avant de cette zone suivi dans le temps.
2. Fine-tuning d'un détecteur « nageur » sur quelques centaines d'images annotées.
