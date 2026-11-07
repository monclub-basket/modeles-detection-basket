// SPDX-License-Identifier: AGPL-3.0-only
// Code publié aussi dans https://github.com/monclub-basket/modeles-detection-basket (dossier suivi/), sous AGPL-3.0,
// parce qu'il porte du code d'ultralytics (AGPL-3.0). Toute modification se fait ici ET là-bas.
/**
 * Suivi ByteTrack (port d'ultralytics BYTETracker et de KalmanFilterXYAH), sans dépendance.
 * Parité mesurée contre ultralytics sur 100 images réelles (bytetrack-parite.json) : mêmes
 * boîtes, mêmes identifiants. Mode « caméra fixe » du worker de télestration ; les caméras
 * mobiles (BoT-SORT, compensation de mouvement) ne sont pas portées.
 */
export type Boite = readonly [number, number, number, number];
export interface Detection { boite: Boite; score: number; /** Classe du détecteur (joueur, ballon…), quand il en rend une. */ classe?: number }
export interface PisteSortie { id: number; boite: Boite }

export interface ReglagesByteTrack {
  seuilHaut: number;
  seuilBas: number;
  seuilNouvellePiste: number;
  tampon: number;
  seuilAppariement: number;
}
const DEFAUTS: ReglagesByteTrack = { seuilHaut: 0.25, seuilBas: 0.1, seuilNouvellePiste: 0.25, tampon: 30, seuilAppariement: 0.8 };

type Matrice = number[][];
const identite = (n: number): Matrice => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
const produit = (a: Matrice, b: Matrice): Matrice => a.map((ligne) => b[0].map((_, j) => ligne.reduce((s, v, k) => s + v * b[k][j], 0)));
const transposee = (a: Matrice): Matrice => a[0].map((_, j) => a.map((ligne) => ligne[j]));
const somme = (a: Matrice, b: Matrice): Matrice => a.map((l, i) => l.map((v, j) => v + b[i][j]));
const difference = (a: Matrice, b: Matrice): Matrice => a.map((l, i) => l.map((v, j) => v - b[i][j]));
const diagonale = (v: number[]): Matrice => v.map((x, i) => v.map((_, j) => (i === j ? x : 0)));

function inverse(a: Matrice): Matrice {
  const n = a.length;
  const m = a.map((ligne, i) => [...ligne, ...identite(n)[i]]);
  for (let c = 0; c < n; c += 1) {
    let pivot = c;
    for (let r = c + 1; r < n; r += 1) if (Math.abs(m[r][c]) > Math.abs(m[pivot][c])) pivot = r;
    [m[c], m[pivot]] = [m[pivot], m[c]];
    const d = m[c][c];
    for (let j = 0; j < 2 * n; j += 1) m[c][j] /= d;
    for (let r = 0; r < n; r += 1) {
      if (r === c) continue;
      const f = m[r][c];
      for (let j = 0; j < 2 * n; j += 1) m[r][j] -= f * m[c][j];
    }
  }
  return m.map((ligne) => ligne.slice(n));
}

const POIDS_POSITION = 1 / 20;
const POIDS_VITESSE = 1 / 160;
const F: Matrice = identite(8);
for (let i = 0; i < 4; i += 1) F[i][i + 4] = 1;
const H: Matrice = Array.from({ length: 4 }, (_, i) => Array.from({ length: 8 }, (_, j) => (i === j ? 1 : 0)));

interface Etat { moyenne: number[]; covariance: Matrice }

function initialiser(z: number[]): Etat {
  const h = z[3];
  const ecarts = [2 * POIDS_POSITION * h, 2 * POIDS_POSITION * h, 1e-2, 2 * POIDS_POSITION * h,
    10 * POIDS_VITESSE * h, 10 * POIDS_VITESSE * h, 1e-5, 10 * POIDS_VITESSE * h];
  return { moyenne: [...z, 0, 0, 0, 0], covariance: diagonale(ecarts.map((e) => e * e)) };
}

function predire(e: Etat): Etat {
  const h = e.moyenne[3];
  const ecarts = [POIDS_POSITION * h, POIDS_POSITION * h, 1e-2, POIDS_POSITION * h,
    POIDS_VITESSE * h, POIDS_VITESSE * h, 1e-5, POIDS_VITESSE * h];
  const bruit = diagonale(ecarts.map((x) => x * x));
  return {
    moyenne: F.map((ligne) => ligne.reduce((s, v, k) => s + v * e.moyenne[k], 0)),
    covariance: somme(produit(produit(F, e.covariance), transposee(F)), bruit),
  };
}

function corriger(e: Etat, z: number[]): Etat {
  const h = e.moyenne[3];
  const ecarts = [POIDS_POSITION * h, POIDS_POSITION * h, 1e-1, POIDS_POSITION * h];
  const R = diagonale(ecarts.map((x) => x * x));
  const S = somme(produit(produit(H, e.covariance), transposee(H)), R);
  const projection = H.map((ligne) => ligne.reduce((s, v, k) => s + v * e.moyenne[k], 0));
  const K = produit(produit(e.covariance, transposee(H)), inverse(S));
  const innovation = z.map((v, i) => v - projection[i]);
  return {
    moyenne: e.moyenne.map((v, i) => v + K[i].reduce((s, kv, k) => s + kv * innovation[k], 0)),
    covariance: difference(e.covariance, produit(produit(K, S), transposee(K))),
  };
}

const versXyah = (b: Boite): number[] => {
  const w = b[2] - b[0];
  const h = b[3] - b[1];
  return [b[0] + w / 2, b[1] + h / 2, w / h, h];
};
const depuisXyah = (m: number[]): Boite => {
  const h = m[3];
  const w = m[2] * h;
  return [m[0] - w / 2, m[1] - h / 2, m[0] + w / 2, m[1] + h / 2];
};

export function iou(a: Boite, b: Boite): number {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]);
  const y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9);
}

/** Algorithme hongrois (matrice carrée). */
function hongrois(cout: number[][]): number[] {
  const n = cout.length;
  const u = Array<number>(n + 1).fill(0);
  const v = Array<number>(n + 1).fill(0);
  const p = Array<number>(n + 1).fill(0);
  const chemin = Array<number>(n + 1).fill(0);
  for (let i = 1; i <= n; i += 1) {
    p[0] = i;
    let j0 = 0;
    const minv = Array<number>(n + 1).fill(Infinity);
    const utilise = Array<boolean>(n + 1).fill(false);
    do {
      utilise[j0] = true;
      const i0 = p[j0];
      let delta = Infinity;
      let j1 = 0;
      for (let j = 1; j <= n; j += 1) {
        if (utilise[j]) continue;
        const courant = cout[i0 - 1][j - 1] - u[i0] - v[j];
        if (courant < minv[j]) { minv[j] = courant; chemin[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= n; j += 1) {
        if (utilise[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta;
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = chemin[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const resultat = Array<number>(n).fill(-1);
  for (let j = 1; j <= n; j += 1) resultat[p[j] - 1] = j - 1;
  return resultat;
}

interface Affectation { paires: Array<[number, number]>; lignesLibres: number[]; colonnesLibres: number[] }

/** Affectation optimale avec seuil de coût (équivalent de lapjv avec cost_limit). */
function affecter(cout: number[][], seuil: number, nColonnes: number): Affectation {
  const n = cout.length;
  const m = nColonnes;
  if (n === 0 || m === 0) {
    return { paires: [], lignesLibres: [...Array(n).keys()], colonnesLibres: [...Array(m).keys()] };
  }
  const N = n + m;
  const GRAND = 1e6;
  const C = Array.from({ length: N }, () => Array<number>(N).fill(GRAND));
  for (let i = 0; i < n; i += 1) for (let j = 0; j < m; j += 1) C[i][j] = cout[i][j] <= seuil ? cout[i][j] : GRAND;
  for (let i = 0; i < n; i += 1) C[i][m + i] = seuil;
  for (let j = 0; j < m; j += 1) C[n + j][j] = seuil;
  for (let i = n; i < N; i += 1) for (let j = m; j < N; j += 1) C[i][j] = 0;
  const r = hongrois(C);
  const paires: Array<[number, number]> = [];
  const lignesLibres: number[] = [];
  const colonnes = new Set<number>([...Array(m).keys()]);
  for (let i = 0; i < n; i += 1) {
    const j = r[i];
    if (j >= 0 && j < m && cout[i][j] <= seuil) { paires.push([i, j]); colonnes.delete(j); } else lignesLibres.push(i);
  }
  return { paires, lignesLibres, colonnesLibres: [...colonnes] };
}

type EtatPiste = 'suivie' | 'perdue' | 'retiree';
interface Piste {
  id: number;
  etat: EtatPiste;
  active: boolean;
  dernier: number;
  kalman: Etat;
  score: number;
}

export interface Suiveur {
  mettreAJour(detections: Detection[]): PisteSortie[];
  /** Oublie toutes les pistes (changement de plan) ; la numérotation repart de 1. */
  reinitialiser(): void;
}

export function creerSuiveur(reglages: Partial<ReglagesByteTrack> = {}): Suiveur {
  const r = { ...DEFAUTS, ...reglages };
  let image = 0;
  let suivies: Piste[] = [];
  let perdues: Piste[] = [];
  let prochainId = 1;

  const boite = (p: Piste): Boite => depuisXyah(p.kalman.moyenne);
  const coutIou = (pistes: Piste[], dets: Detection[]) => pistes.map((p) => dets.map((d) => 1 - iou(boite(p), d.boite)));
  const fusionner = (cout: number[][], dets: Detection[]) => cout.map((l) => l.map((v, j) => 1 - (1 - v) * dets[j].score));
  const appliquer = (p: Piste, d: Detection) => {
    p.kalman = corriger(p.kalman, versXyah(d.boite));
    p.score = d.score;
    p.etat = 'suivie';
    p.active = true;
    p.dernier = image;
  };

  return {
    reinitialiser() {
      image = 0;
      suivies = [];
      perdues = [];
      prochainId = 1;
    },
    mettreAJour(detections) {
      image += 1;
      const hautes = detections.filter((d) => d.score >= r.seuilHaut);
      const secondes = detections.filter((d) => d.score > r.seuilBas && d.score < r.seuilHaut);
      const nonConfirmees = suivies.filter((p) => !p.active);
      const actives = suivies.filter((p) => p.active);
      const reserve = [...actives];
      for (const p of perdues) if (!reserve.includes(p)) reserve.push(p);
      for (const p of reserve) {
        if (p.etat !== 'suivie') p.kalman.moyenne[7] = 0;
        p.kalman = predire(p.kalman);
      }

      const a1 = affecter(fusionner(coutIou(reserve, hautes), hautes), r.seuilAppariement, hautes.length);
      const miseAJour: Piste[] = [];
      for (const [i, j] of a1.paires) { appliquer(reserve[i], hautes[j]); miseAJour.push(reserve[i]); }

      const restantes = a1.lignesLibres.map((i) => reserve[i]).filter((p) => p.etat === 'suivie');
      const a2 = affecter(coutIou(restantes, secondes), 0.5, secondes.length);
      for (const [i, j] of a2.paires) { appliquer(restantes[i], secondes[j]); miseAJour.push(restantes[i]); }
      const nouvellesPerdues: Piste[] = [];
      for (const i of a2.lignesLibres) {
        const p = restantes[i];
        if (p.etat !== 'perdue') { p.etat = 'perdue'; nouvellesPerdues.push(p); }
      }

      const restesHautes = a1.colonnesLibres.map((j) => hautes[j]);
      const a3 = affecter(fusionner(coutIou(nonConfirmees, restesHautes), restesHautes), 0.7, restesHautes.length);
      for (const [i, j] of a3.paires) { appliquer(nonConfirmees[i], restesHautes[j]); miseAJour.push(nonConfirmees[i]); }
      for (const i of a3.lignesLibres) nonConfirmees[i].etat = 'retiree';

      const utilisees = new Set(a3.paires.map((p) => p[1]));
      const neuves: Piste[] = [];
      restesHautes.forEach((d, j) => {
        if (utilisees.has(j) || d.score < r.seuilNouvellePiste) return;
        neuves.push({ id: prochainId++, etat: 'suivie', active: image === 1, dernier: image, kalman: initialiser(versXyah(d.boite)), score: d.score });
      });

      for (const p of perdues) if (image - p.dernier > r.tampon) p.etat = 'retiree';
      suivies = [...new Set([...suivies.filter((p) => p.etat === 'suivie'), ...miseAJour, ...neuves])].filter((p) => p.etat === 'suivie');
      perdues = [...new Set([...perdues.filter((p) => p.etat === 'perdue' && !suivies.includes(p)), ...nouvellesPerdues])]
        .filter((p) => p.etat === 'perdue' && image - p.dernier <= r.tampon);
      return suivies.filter((p) => p.active).map((p) => ({ id: p.id, boite: boite(p) }));
    },
  };
}
