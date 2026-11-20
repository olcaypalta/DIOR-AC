import argparse
import csv
import time
import cv2
import numpy as np
import degerlendir as D
from bozulma import _smooth_noise
from onarim import satir_doldur

def blok_kayip(x, p, rng, blok=16):
    h = x.shape[0]
    out = x.copy()
    for b0 in range(0, h, blok):
        if rng.random() < p:
            out[b0:b0 + blok] = 0
    return out

def hafif_gurultu(x, lam, rng):
    J = x.astype(np.float32) / 255
    n = rng.poisson(J * lam) / lam + rng.normal(0, 0.005, J.shape)
    return np.clip(n * 255, 0, 255).astype(np.uint8)

def agir_pus(x, t, rng):
    A = 0.85 + 0.1 * rng.random()
    J = x.astype(np.float32) / 255
    return np.clip((J * t + A * (1 - t)) * 255, 0, 255).astype(np.uint8)

def agir_bulut(x, op, rng):
    h, w = x.shape[:2]
    a = _smooth_noise(h, w, 96, rng) * op
    J = x.astype(np.float32) / 255
    return np.clip((J * (1 - a[..., None]) + 0.95 * a[..., None]) * 255, 0, 255).astype(np.uint8)
KOSUL = [('block_dropout', p, lambda x, r, p=p: blok_kayip(x, p, r)) for p in (0.03, 0.08, 0.15)] + [('block_dropout_interp', p, lambda x, r, p=p: satir_doldur(blok_kayip(x, p, r))) for p in (0.03, 0.08, 0.15)] + [('sensor_noise_light', lam, lambda x, r, lam=lam: hafif_gurultu(x, lam, r)) for lam in (1000, 200)] + [('haze_heavy', t, lambda x, r, t=t: agir_pus(x, t, r)) for t in (0.25, 0.15)] + [('thin_cloud_heavy', op, lambda x, r, op=op: agir_bulut(x, op, r)) for op in (0.9, 1.0)]

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--model', required=True)
    a.add_argument('--tag', required=True)
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--n', type=int, default=500)
    a.add_argument('--veri', default='/workspace/dior')
    a.add_argument('--out', default='/workspace/results')
    g = a.parse_args()
    gt, imgs = D.veri_yukle(g.veri, g.n, g.seed)
    model = D.model_kur(g.model)
    with open(f'{g.out}/{g.tag}_ek.csv', 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'seed', 'n', 'corruption', 'param', 'mAP', 'mAP50', 'AR100', 'sec'])
        for k, (ad, par, fn) in enumerate(KOSUL):
            t0 = time.time()
            rng = np.random.default_rng(1000 * g.seed + 7919 + k)
            tahmin = []
            for i, img in enumerate(imgs):
                kutu, skor, lab = model(fn(img, rng))
                for b, sc, l in zip(kutu, skor, lab):
                    tahmin.append({'image_id': i, 'category_id': int(l), 'bbox': [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], 'score': float(sc)})
            m, m50, ar = D.degerlendir(gt, tahmin)
            w.writerow([g.tag, g.seed, g.n, ad, par, f'{m:.2f}', f'{m50:.2f}', f'{ar:.2f}', f'{time.time() - t0:.0f}'])
            f.flush()
            print(g.tag, ad, par, f'mAP={m:.2f}', flush=True)
if __name__ == '__main__':
    main()
