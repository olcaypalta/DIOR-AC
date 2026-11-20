import argparse
import csv
import glob
import gzip
import io
import json
import os
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from bozulma import BOZULMALAR, uygula
SINIFLAR = ['airplane', 'airport', 'baseball field', 'basketball court', 'bridge', 'chimney', 'dam', 'expressway service area', 'expressway toll station', 'golf course', 'ground track field', 'harbor', 'overpass', 'ship', 'stadium', 'storage tank', 'tennis court', 'train station', 'vehicle', 'windmill']

def veri_yukle(kok, n, seed):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f'{kok}/data/test-*.parquet'))])
    df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    resimler, anns = ([], [])
    for i, r in df.iterrows():
        resimler.append({'id': i, 'width': int(r.width), 'height': int(r.height)})
        o = r.objects
        for b, c in zip(o['bbox'], o['category']):
            x0, y0, bw, bh = [float(v) for v in b]
            anns.append({'id': len(anns) + 1, 'image_id': i, 'category_id': int(c), 'bbox': [x0, y0, bw, bh], 'area': bw * bh, 'iscrowd': 0})
    gt = {'images': resimler, 'annotations': anns, 'categories': [{'id': k, 'name': s} for k, s in enumerate(SINIFLAR)]}
    goruntuler = [np.array(Image.open(io.BytesIO(r.image['bytes'])).convert('RGB')) for _, r in df.iterrows()]
    return (gt, goruntuler)

class OWLv2:

    def __init__(self):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.p = Owlv2Processor.from_pretrained('google/owlv2-base-patch16-ensemble')
        self.m = Owlv2ForObjectDetection.from_pretrained('google/owlv2-base-patch16-ensemble').cuda().eval().half()

    @torch.no_grad()
    def __call__(self, img):
        h, w = img.shape[:2]
        x = self.p(text=[[f'a satellite photo of a {s}' for s in SINIFLAR]], images=img, return_tensors='pt').to('cuda')
        x['pixel_values'] = x['pixel_values'].half()
        o = self.m(**x)
        s = max(h, w)
        r = self.p.post_process_object_detection(o, threshold=0.05, target_sizes=torch.tensor([[s, s]]).cuda())[0]
        return (r['boxes'].float().cpu().numpy(), r['scores'].float().cpu().numpy(), r['labels'].cpu().numpy())

class GDINO:

    def __init__(self):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        mid = 'IDEA-Research/grounding-dino-tiny'
        self.p = AutoProcessor.from_pretrained(mid)
        self.m = AutoModelForZeroShotObjectDetection.from_pretrained(mid).cuda().eval()
        self.metin = '. '.join(SINIFLAR) + '.'

    @torch.no_grad()
    def __call__(self, img):
        h, w = img.shape[:2]
        x = self.p(images=img, text=self.metin, return_tensors='pt').to('cuda')
        o = self.m(**x)
        r = self.p.post_process_grounded_object_detection(o, x.input_ids, threshold=0.1, text_threshold=0.1, target_sizes=[(h, w)])[0]
        etiket = r.get('text_labels', r.get('labels'))
        lab = []
        for e in etiket:
            e = str(e).strip()
            eslesen = [k for k, s in enumerate(SINIFLAR) if s == e] or [k for k, s in enumerate(SINIFLAR) if e and (e in s or s in e)]
            lab.append(eslesen[0] if eslesen else -1)
        lab = np.array(lab)
        tut = lab >= 0
        return (r['boxes'].cpu().numpy()[tut], r['scores'].cpu().numpy()[tut], lab[tut])

class YOLOWorld:

    def __init__(self):
        from ultralytics import YOLOWorld as Y
        self.m = Y('yolov8l-worldv2.pt')
        self.m.set_classes(SINIFLAR)

    def __call__(self, img):
        r = self.m.predict(img[..., ::-1], conf=0.01, verbose=False, imgsz=800)[0]
        b = r.boxes
        return (b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int))

class YOLOFT:

    def __init__(self, yol):
        from ultralytics import YOLO
        self.m = YOLO(yol)

    def __call__(self, img):
        r = self.m.predict(img[..., ::-1], conf=0.001, verbose=False, imgsz=800, max_det=300)[0]
        b = r.boxes
        return (b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int))
MODELLER = {'owlv2': OWLv2, 'gdino': GDINO, 'yoloworld': YOLOWorld}

def model_kur(ad):
    return YOLOFT(ad[3:]) if ad.startswith('ft:') else MODELLER[ad]()

def degerlendir(gt, tahmin):
    coco = COCO()
    coco.dataset = gt
    coco.createIndex()
    if not tahmin:
        return (0.0, 0.0, 0.0)
    dt = coco.loadRes(tahmin)
    e = COCOeval(coco, dt, 'bbox')
    e.params.maxDets = [1, 10, 100]
    e.evaluate()
    e.accumulate()
    e.summarize()
    return (e.stats[0] * 100, e.stats[1] * 100, e.stats[8] * 100)

def main():
    a = argparse.ArgumentParser()
    a.add_argument('--model', required=True)
    a.add_argument('--seed', type=int, default=0)
    a.add_argument('--n', type=int, default=300)
    a.add_argument('--veri', default='/workspace/dior')
    a.add_argument('--out', default='/workspace/results')
    a.add_argument('--tag', default=None)
    g = a.parse_args()
    os.makedirs(g.out, exist_ok=True)
    etiket = g.tag or g.model
    yol = f'{g.out}/{etiket}_s{g.seed}.csv' if not g.tag else f'{g.out}/{etiket}.csv'
    bitti = set()
    if os.path.exists(yol):
        bitti = {(r['corruption'], int(r['severity'])) for r in csv.DictReader(open(yol))}
    gt, imgs = veri_yukle(g.veri, g.n, g.seed)
    model = model_kur(g.model)
    kosullar = [('clean', 0)] + [(b, s) for b in BOZULMALAR for s in (1, 2, 3)]
    yeni = not os.path.exists(yol)
    with open(yol, 'a', newline='') as f:
        w = csv.writer(f)
        if yeni:
            w.writerow(['model', 'seed', 'n', 'corruption', 'severity', 'mAP', 'mAP50', 'AR100', 'sec'])
        for ad, s in kosullar:
            if (ad, s) in bitti:
                continue
            t0 = time.time()
            rng = np.random.default_rng(1000 * g.seed + 17 * s + sum(map(ord, ad)))
            tahmin = []
            for i, img in enumerate(imgs):
                kutu, skor, lab = model(uygula(img, ad, s, rng))
                for b, sc, l in zip(kutu, skor, lab):
                    tahmin.append({'image_id': i, 'category_id': int(l), 'bbox': [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])], 'score': float(sc)})
            pdir = f'{g.out}/pred/{etiket}_s{g.seed}'
            os.makedirs(pdir, exist_ok=True)
            if not os.path.exists(f'{pdir}/gt.json.gz'):
                with gzip.open(f'{pdir}/gt.json.gz', 'wt') as z:
                    json.dump(gt, z)
            with gzip.open(f'{pdir}/{ad}_{s}.json.gz', 'wt') as z:
                json.dump(tahmin, z)
            m, m50, ar = degerlendir(gt, tahmin)
            w.writerow([etiket, g.seed, g.n, ad, s, f'{m:.2f}', f'{m50:.2f}', f'{ar:.2f}', f'{time.time() - t0:.0f}'])
            f.flush()
            print(etiket, g.seed, ad, s, f'mAP={m:.2f} mAP50={m50:.2f}', flush=True)
if __name__ == '__main__':
    main()
