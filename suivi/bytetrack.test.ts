import { describe, expect, it } from 'vitest';
import fixture from './fixtures/bytetrack-parite.json';
import { creerSuiveur, iou, type Detection } from './bytetrack';

const det = (x: number, y: number, w = 40, h = 100, s = 0.9): Detection => ({ boite: [x, y, x + w, y + h], score: s });

describe('suivi ByteTrack : parité avec ultralytics', () => {
  it('rejoue 100 images réelles : mêmes boîtes, mêmes identifiants que le worker Python', () => {
    const suiveur = creerSuiveur();
    let total = 0;
    let apparies = 0;
    const correspondance = new Map<string, number>();
    fixture.dets.forEach((image, i) => {
      const sorties = suiveur.mettreAJour(image.map((d) => ({ boite: [d[0], d[1], d[2], d[3]] as const, score: d[4] })));
      for (const sortie of sorties) {
        let meilleure: number[] | null = null;
        let meilleurIou = 0;
        for (const p of fixture.trk[i]) {
          const v = iou(sortie.boite, [p[0], p[1], p[2], p[3]]);
          if (v > meilleurIou) { meilleurIou = v; meilleure = p; }
        }
        total += 1;
        if (meilleure && meilleurIou > 0.5) {
          apparies += 1;
          const cle = `${sortie.id}>${meilleure[4]}`;
          correspondance.set(cle, (correspondance.get(cle) ?? 0) + 1);
        }
      }
    });
    expect(total).toBe(572);
    expect(apparies).toBe(total);
    const parIdTs = new Map<string, Map<string, number>>();
    for (const [cle, n] of correspondance) {
      const [a, b] = cle.split('>');
      parIdTs.set(a, (parIdTs.get(a) ?? new Map()).set(b, n));
    }
    // Un identifiant du port correspond toujours au même identifiant ultralytics.
    for (const cibles of parIdTs.values()) expect(cibles.size).toBe(1);
    expect(parIdTs.size).toBe(new Set(fixture.trk.flat().map((p) => p[4])).size);
  });
});

describe('suivi ByteTrack : cycle de vie des pistes', () => {
  it('garde l’identifiant d’un joueur qui avance', () => {
    const s = creerSuiveur();
    const ids = new Set<number>();
    for (let i = 0; i < 20; i += 1) for (const o of s.mettreAJour([det(100 + i * 3, 200)])) ids.add(o.id);
    expect([...ids]).toEqual([1]);
  });

  it('une piste née après la première image n’est sortie qu’à sa deuxième détection', () => {
    const s = creerSuiveur();
    s.mettreAJour([det(100, 200)]);
    expect(s.mettreAJour([det(102, 200), det(600, 300)]).map((o) => o.id)).toEqual([1]);
    expect(s.mettreAJour([det(104, 200), det(603, 300)]).map((o) => o.id).sort()).toEqual([1, 2]);
  });

  it('retrouve le même identifiant après une courte disparition (tampon de 30 images)', () => {
    const s = creerSuiveur();
    for (let i = 0; i < 5; i += 1) s.mettreAJour([det(100 + i * 2, 200)]);
    for (let i = 0; i < 10; i += 1) expect(s.mettreAJour([])).toEqual([]);
    const retour = s.mettreAJour([det(120, 200)]);
    expect(retour.map((o) => o.id)).toEqual([1]);
  });

  it('ne réutilise pas une piste perdue depuis plus de 30 images : nouvel identifiant', () => {
    const s = creerSuiveur();
    for (let i = 0; i < 5; i += 1) s.mettreAJour([det(100, 200)]);
    for (let i = 0; i < 40; i += 1) s.mettreAJour([]);
    s.mettreAJour([det(100, 200)]);
    expect(s.mettreAJour([det(101, 200)]).map((o) => o.id)).toEqual([2]);
  });

  it('une réinitialisation (coupe) oublie toutes les pistes', () => {
    const s = creerSuiveur();
    for (let i = 0; i < 3; i += 1) s.mettreAJour([det(100, 200)]);
    s.reinitialiser();
    expect(s.mettreAJour([det(100, 200)]).map((o) => o.id)).toEqual([1]);
  });
});
