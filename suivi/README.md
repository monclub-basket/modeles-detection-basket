# Suivi des joueurs (ByteTrack, caméra fixe)

Suiveur d'objets en TypeScript, sans dépendance, utilisé par la coque de bureau de MonClubBasket pour attribuer des identifiants stables aux joueurs détectés image par image. Publié par le club MCBB sous licence **GNU AGPL-3.0** (voir `../LICENSE`).

| Fichier | Contenu |
|---|---|
| `bytetrack.ts` | association en deux passes (ByteTrack) et filtre de Kalman à état `xyah` |
| `bytetrack.test.ts` | tests, dont la comparaison aux pistes de référence |
| `fixtures/bytetrack-parite.json` | 100 images réelles (caméra fixe 1920×844, 25 i/s) : détections du modèle gymnase (confiance ≥ 0,2, classes 3 à 7) et pistes de référence produites par `ultralytics` (`bytetrack.yaml`). Des nombres seulement, aucune image. |

## Provenance
`bytetrack.ts` est un **port en TypeScript** de `BYTETracker` et de `KalmanFilterXYAH` d'Ultralytics 8.4.137 (AGPL-3.0). Il en dérive donc et est distribué sous AGPL-3.0, comme le demande cette licence. L'algorithme d'origine est ByteTrack (Zhang et al., 2022).

Sur les 100 images de `fixtures/`, il rend les mêmes boîtes et les mêmes identifiants que la référence.

## Limites
- Mode caméra fixe seulement. Les caméras mobiles (BoT-SORT, compensation de mouvement) ne sont pas portées.
- La coque qui l'utilise contient d'autres fichiers qui ne sont pas publiés ici. Ce dépôt ne dit rien de juridique sur l'étendue de l'AGPL pour ces autres fichiers.

## Lancer les tests
Avec `vitest` : `npx vitest run suivi/bytetrack.test.ts` (le test lit `./fixtures/bytetrack-parite.json`).
