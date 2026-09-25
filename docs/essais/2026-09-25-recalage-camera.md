# Essai — recalage d'une caméra qui suit la course (2026-09-25)

## Vidéo

50 NL China Open 2026 filmé au téléphone depuis les gradins, bassin de 50 m, la
caméra tourne d'environ 90° pour suivre la course du départ à l'arrivée (`data/`).

## Essais

1. **Chaînage d'homographies image → image de référence** (SIFT 3000 points,
   ratio de Lowe 0,75, RANSAC 2 px, image clé renouvelée sous 150 points cohérents),
   une image sur trois, 960×540.
   - Correspondances solides tout du long : 150 à 1 450 points cohérents (médiane 513).
   - Mais projeter sur le **plan de la première image** diverge au-delà de ~60° de
     rotation (largeur projetée ×100 à t = 20 s) : limite géométrique d'un panorama
     plan, pas une erreur de recalage.
2. **Panorama sphérique OpenCV** (`Stitcher_PANORAMA`) sur 14 images de 0,5 s à 26,5 s :
   succès. Le bassin de 50 m entier est reconstitué, lignes d'eau continues du départ
   à l'arrivée.

## Conclusion

Le recalage sur le décor fixe (gradins, bords, lignes du fond) fonctionne sur une
vidéo réelle qui panoramique. Il faut projeter directement sur le **plan du bassin**
(`H_bassin←t = H_bassin←réf · H_réf←t`, en coordonnées homogènes), jamais sur le plan
d'une image. Reste à mesurer la dérive du chaînage en mètres, une fois l'image de
référence calibrée.
