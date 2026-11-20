import argparse
import csv
import os
import time
from multiprocessing import Pool
import degerlendir as D
from onarim import ONARIM

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
    with open(f'{g.out}/{g.tag}_onarim_temiz.csv', 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'seed', 'n', 'corruption', 'severity', 'repair', 'mAP', 'mAP50', 'AR100', 'sec'])
        for fn in ONARIM.values():
            t0 = time.time()
            with Pool(min(12, os.cpu_count() or 1)) as havuz:
                onarilmis = havuz.map(fn, imgs, chunksize=8)
            tahmin = []
            for i, img in enumerate(onarilmis):
                kutu, skor, lab = model(img)
                for b, sc, l in zip(kutu, skor, lab):
                    tahmin.append({'image_id': i, 'category_id': int(l), 'bbox': [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], 'score': float(sc)})
            m, m50, ar = D.degerlendir(gt, tahmin)
            w.writerow([g.tag, g.seed, g.n, 'clean', 0, fn.__name__, f'{m:.2f}', f'{m50:.2f}', f'{ar:.2f}', f'{time.time() - t0:.0f}'])
            f.flush()
            print(g.tag, 'clean', fn.__name__, f'mAP={m:.2f}', flush=True)
if __name__ == '__main__':
    main()
