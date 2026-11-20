import argparse
import glob
import io
import os
import sys
import pandas as pd
from PIL import Image

def yolo_veri(kok, hedef):
    if os.path.exists(f'{hedef}/dior.yaml'):
        return f'{hedef}/dior.yaml'
    from degerlendir import SINIFLAR
    for bolme, desen in (('train', 'train-*'), ('val', 'validation-*')):
        os.makedirs(f'{hedef}/images/{bolme}', exist_ok=True)
        os.makedirs(f'{hedef}/labels/{bolme}', exist_ok=True)
        for f in sorted(glob.glob(f'{kok}/data/{desen}.parquet')):
            for _, r in pd.read_parquet(f).iterrows():
                ad = f'{int(r.image_id):06d}'
                Image.open(io.BytesIO(r.image['bytes'])).convert('RGB').save(f'{hedef}/images/{bolme}/{ad}.jpg', quality=95)
                w, h = (float(r.width), float(r.height))
                with open(f'{hedef}/labels/{bolme}/{ad}.txt', 'w') as t:
                    for b, c in zip(r.objects['bbox'], r.objects['category']):
                        x, y, bw, bh = [float(v) for v in b]
                        t.write(f'{int(c)} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}\n')
    with open(f'{hedef}/dior.yaml', 'w') as y:
        y.write(f'path: {hedef}\ntrain: images/train\nval: images/val\nnames:\n')
        for k, s in enumerate(SINIFLAR):
            y.write(f'  {k}: {s}\n')
    return f'{hedef}/dior.yaml'

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--arch', default='yolo11s')
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--epochs', type=int, default=12)
    a.add_argument('--n', type=int, default=500)
    a.add_argument('--veri', default='/workspace/dior')
    a.add_argument('--out', default='/workspace/results')
    g = a.parse_args()
    from ultralytics import YOLO
    yaml = yolo_veri(g.veri, '/workspace/dior_yolo')
    ad = f'{g.arch}_s{g.seed}'
    agirlik = f'/workspace/runs/{ad}/weights/best.pt'
    if not os.path.exists(agirlik):
        m = YOLO(f'{g.arch}.pt')
        son = f'/workspace/runs/{ad}/weights/last.pt'
        if os.path.exists(son):
            YOLO(son).train(resume=True)
        else:
            m.train(data=yaml, epochs=g.epochs, imgsz=800, batch=16, seed=g.seed, deterministic=False, project='/workspace/runs', name=ad, exist_ok=True, workers=8, plots=False)
    sys.argv = ['degerlendir.py', '--model', f'ft:{agirlik}', '--seed', str(g.seed), '--n', str(g.n), '--veri', g.veri, '--out', g.out, '--tag', ad]
    import degerlendir
    degerlendir.main()
if __name__ == '__main__':
    main()
