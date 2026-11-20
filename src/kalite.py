import glob
import io
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from bozulma import BOZULMALAR, uygula
KOK = Path(__file__).resolve().parents[1]

def ssim(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = ((0.01 * 255) ** 2, (0.03 * 255) ** 2)
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = (g(a), g(b))
    va, vb, cab = (g(a * a) - ma ** 2, g(b * b) - mb ** 2, g(a * b) - ma * mb)
    return float(((2 * ma * mb + c1) * (2 * cab + c2) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean())

def psnr(a, b):
    m = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if m == 0 else float(10 * np.log10(255 ** 2 / m))

def main(n=500, seed=0):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(KOK / 'veri' / 'test-*.parquet')))])
    df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    imgs = [np.array(Image.open(io.BytesIO(r.image['bytes'])).convert('RGB')) for _, r in df.iterrows()]
    satir = []
    for ad in BOZULMALAR:
        for s in (1, 2, 3):
            rng = np.random.default_rng(1000 * seed + 17 * s + sum(map(ord, ad)))
            p, q = ([], [])
            for x in imgs:
                y = uygula(x, ad, s, rng)
                p.append(psnr(x, y))
                q.append(ssim(x, y))
            satir.append({'corruption': ad, 'severity': s, 'psnr': np.mean(p), 'ssim': np.mean(q)})
            print(ad, s, round(np.mean(p), 2), round(np.mean(q), 3), flush=True)
    pd.DataFrame(satir).to_csv(KOK / 'results' / 'tablolar' / 't5_kalite.csv', index=False)
if __name__ == '__main__':
    main(*[int(a) for a in sys.argv[1:]])
