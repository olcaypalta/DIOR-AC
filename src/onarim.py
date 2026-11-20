import argparse
import csv
import os
import time
from multiprocessing import Pool
import cv2
import numpy as np
import degerlendir as D
from bozulma import uygula

def satir_doldur(y):
    eksik = np.all(y == 0, axis=(1, 2))
    if not eksik.any():
        return y
    z = y.astype(np.float32)
    var = np.where(~eksik)[0]
    for k in range(3):
        for j in range(y.shape[1]):
            z[eksik, j, k] = np.interp(np.where(eksik)[0], var, z[var, j, k])
    return np.clip(z, 0, 255).astype(np.uint8)

def serit_gider(y):
    z = y.astype(np.float32)
    m, s = (z.mean(axis=0, keepdims=True), z.std(axis=0, keepdims=True) + 1e-06)
    gm, gs = (z.mean(axis=(0, 1), keepdims=True), z.std(axis=(0, 1), keepdims=True))
    return np.clip((z - m) / s * gs + gm, 0, 255).astype(np.uint8)

def gurultu_gider(y):
    return cv2.fastNlMeansDenoisingColored(y, None, 10, 10, 7, 21)
ONARIM = {'line_dropout': satir_doldur, 'stripe_noise': serit_gider, 'sensor_noise': gurultu_gider}

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--model', required=True)
    a.add_argument('--tag', required=True)
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--n', type=int, default=500)
    a.add_argument('--veri', default='/workspace/dior')
    a.add_argument('--out', default='/workspace/results')
    g = a.parse_args()
    os.makedirs(g.out, exist_ok=True)
    yol = f'{g.out}/{g.tag}_onarim.csv'
    gt, imgs = D.veri_yukle(g.veri, g.n, g.seed)
    model = D.model_kur(g.model)
    with open(yol, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'seed', 'n', 'corruption', 'severity', 'repair', 'mAP', 'mAP50', 'AR100', 'sec'])
        for ad, fn in ONARIM.items():
            for s in (1, 2, 3):
                t0 = time.time()
                rng = np.random.default_rng(1000 * g.seed + 17 * s + sum(map(ord, ad)))
                tahmin = []
                bozuk = [uygula(img, ad, s, rng) for img in imgs]
                with Pool(min(12, os.cpu_count() or 1)) as havuz:
                    onarilmis = havuz.map(fn, bozuk, chunksize=8)
                for i, img in enumerate(onarilmis):
                    kutu, skor, lab = model(img)
                    for b, sc, l in zip(kutu, skor, lab):
                        tahmin.append({'image_id': i, 'category_id': int(l), 'bbox': [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], 'score': float(sc)})
                m, m50, ar = D.degerlendir(gt, tahmin)
                w.writerow([g.tag, g.seed, g.n, ad, s, fn.__name__, f'{m:.2f}', f'{m50:.2f}', f'{ar:.2f}', f'{time.time() - t0:.0f}'])
                f.flush()
                print(g.tag, ad, s, fn.__name__, f'mAP={m:.2f}', flush=True)
if __name__ == '__main__':
    main()
