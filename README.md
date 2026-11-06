# Modèles de détection pour le basket (MCBB)

Poids d'un détecteur d'objets pour le basket (joueur, joueur avec ballon, tir, arbitre, panier, ballon, numéro), publiés par le club MCBB sous licence **GNU AGPL-3.0** (voir `LICENSE`).

| Fichier | Contenu | SHA-256 |
|---|---|---|
| `poids/basketball_v3_full.pt` | checkpoint PyTorch (Ultralytics 8.4.137), entraîné à 640 px | voir `SHA256SUMS` |
| `poids/basketball_v3_full_1280.onnx` | export ONNX, entrée 1280×1280, NMS intégrée, sortie `[1,300,6]` (x1, y1, x2, y2, score, classe) | voir `SHA256SUMS` |

Classes : 0 ball, 1 ball-in-basket, 2 number, 3 player, 4 player-in-possession, 5 player-jump-shot, 6 player-layup-dunk, 7 player-shot-block, 8 referee, 9 rim.

## Provenance
1. Point de départ : `yolov8n.pt` (Ultralytics, AGPL-3.0).
2. Entraînement `basketball_warmstart` (100 époques, 640 px) puis `basketball_v3_full` (60 époques, 640 px) avec Ultralytics 8.4.137, sur le seul jeu public **basketball-player-detection-3** (654 images, Roboflow Universe, licence **CC BY 4.0**) : https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo . Attribution : jeu fourni par un utilisateur de Roboflow.
3. Aucune image du club, aucune image de personne identifiable propre au club, aucun modèle ni sortie d'un produit tiers n'a servi à l'entraînement.

## Licence
Ces poids sont dérivés d'un modèle Ultralytics et sont donc distribués sous AGPL-3.0, comme l'exige la licence d'Ultralytics. Quiconque les redistribue ou les fait fonctionner pour des tiers à travers un réseau doit respecter l'AGPL-3.0 pour ces fichiers (dont la mise à disposition du code correspondant et de cette provenance). Ce dépôt ne dit rien de juridique sur les logiciels qui chargent ces fichiers comme données.
