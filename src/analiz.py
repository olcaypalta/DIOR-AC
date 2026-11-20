import glob
import gzip
import json
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
KOK = Path(__file__).resolve().parents[1]
RENK = {'yolo11m': '#4472C4', 'yolo11s': '#70AD47', 'owlv2': '#ED7D31', 'gdino': '#FFC000', 'yoloworld': '#A5A5A5'}
AD = {'yolo11m': 'YOLO11m', 'yolo11s': 'YOLO11s', 'owlv2': 'OWLv2', 'gdino': 'Grounding DINO', 'yoloworld': 'YOLO-World'}
SIRA = ['yolo11m', 'yolo11s', 'owlv2', 'gdino', 'yoloworld']
BOZ = ['haze', 'thin_cloud', 'stripe_noise', 'sensor_noise', 'line_dropout', 'jpeg2000', 'mtf_blur', 'low_sun']
BAD = {'en': {'haze': 'Haze', 'thin_cloud': 'Thin cloud', 'stripe_noise': 'Stripe noise', 'sensor_noise': 'Sensor noise', 'line_dropout': 'Line dropout', 'jpeg2000': 'JPEG2000', 'mtf_blur': 'MTF blur', 'low_sun': 'Low sun'}, 'tr': {'haze': 'Pus', 'thin_cloud': 'İnce bulut', 'stripe_noise': 'Şerit gürültüsü', 'sensor_noise': 'Sensör gürültüsü', 'line_dropout': 'Satır kaybı', 'jpeg2000': 'JPEG2000', 'mtf_blur': 'MTF bulanıklığı', 'low_sun': 'Düşük güneş'}}
YAZI = {'en': {'sev': 'Severity', 'map': 'mAP (%)', 'rpc': 'rPC (%)', 'clean': 'clean'}, 'tr': {'sev': 'Şiddet', 'map': 'mAP (%)', 'rpc': 'rPC (%)', 'clean': 'temiz'}}

def oku():
    dfs = [pd.read_csv(f) for f in glob.glob(str(KOK / 'results' / '*' / 'results' / '*.csv'))]
    df = pd.concat(dfs, ignore_index=True)
    tekrar = df[df.model.str.endswith('_pred')].copy()
    df = df[~df.model.str.endswith('_pred')].copy()
    if not tekrar.empty:
        tekrar['model'] = tekrar.model.str.replace('_pred', '', regex=False)
        k = df.merge(tekrar, on=['model', 'seed', 'corruption', 'severity'], suffixes=('', '_t'))
        fark = (k.mAP - k.mAP_t).abs().max()
        print(f'tekrar koşusu: {len(k)} koşul, en büyük mAP farkı {fark:.3f}')
    df['arch'] = df['model'].str.replace('_s\\d+$', '', regex=True)
    return df.drop_duplicates(['arch', 'seed', 'corruption', 'severity'], keep='last')

def ozet(df):
    satir = []
    for (a, s), g in df.groupby(['arch', 'seed']):
        temiz = g[g.corruption == 'clean']
        boz = g[g.corruption != 'clean']
        if temiz.empty or len(boz) < 24:
            continue
        c = float(temiz.mAP.iloc[0])
        satir.append({'arch': a, 'seed': s, 'clean': c, 'clean50': float(temiz.mAP50.iloc[0]), 'mPC': boz.mAP.mean(), 'rPC': 100 * boz.mAP.mean() / c, 'n_kosul': len(boz)})
    return pd.DataFrame(satir)

def ms(x):
    return f'{x.mean():.2f} ± {x.std(ddof=1):.2f}' if len(x) > 1 else f'{x.mean():.2f}'

def tablolar(df, oz):
    out = KOK / 'results' / 'tablolar'
    out.mkdir(parents=True, exist_ok=True)
    t1 = oz.groupby('arch').agg(n=('seed', 'count'), clean=('clean', ms), clean50=('clean50', ms), mPC=('mPC', ms), rPC=('rPC', ms)).reindex([a for a in SIRA if a in set(oz.arch)])
    t1.to_csv(out / 't1_ana.csv')
    bz = df[df.corruption != 'clean'].groupby(['arch', 'seed', 'corruption']).mAP.mean().reset_index()
    t2 = bz.groupby(['arch', 'corruption']).mAP.agg(ms).unstack().reindex(columns=BOZ)
    t2.to_csv(out / 't2_bozulma_map.csv')
    tm = df[df.corruption == 'clean'][['arch', 'seed', 'mAP']].rename(columns={'mAP': 'c'})
    r = bz.merge(tm, on=['arch', 'seed'])
    r['rpc'] = 100 * r.mAP / r.c
    t3 = r.groupby(['arch', 'corruption']).rpc.agg(ms).unstack().reindex(columns=BOZ)
    t3.to_csv(out / 't3_bozulma_rpc.csv')
    print(t1.to_string(), '\n', t3.to_string())
    return r

def siddet_egrisi(df, dil):
    y = YAZI[dil]
    fig, ax = plt.subplots(2, 4, figsize=(12, 5.6), sharey=False)
    for k, b in enumerate(BOZ):
        a = ax.flat[k]
        for m in [m for m in SIRA if m in set(df.arch)]:
            g = df[df.arch == m]
            c = g[g.corruption == 'clean'].groupby('seed').mAP.first()
            ss = [c] + [g[(g.corruption == b) & (g.severity == s)].groupby('seed').mAP.first() for s in (1, 2, 3)]
            mu = np.array([v.mean() for v in ss])
            sd = np.array([v.std(ddof=1) if len(v) > 1 else 0 for v in ss])
            a.plot([0, 1, 2, 3], mu, '-o', ms=3.5, lw=1.6, color=RENK[m], label=AD[m])
            a.fill_between([0, 1, 2, 3], mu - sd, mu + sd, color=RENK[m], alpha=0.18, lw=0)
        a.set_title(BAD[dil][b], fontsize=10)
        a.set_xticks([0, 1, 2, 3], [y['clean'], '1', '2', '3'], fontsize=8)
        a.tick_params(labelsize=8)
        a.grid(alpha=0.3, lw=0.5)
        if k % 4 == 0:
            a.set_ylabel(y['map'], fontsize=9)
        if k >= 4:
            a.set_xlabel(y['sev'], fontsize=9)
    h, l = ax.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=len(l), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    kaydet(fig, 'fig_severity', dil)

def isi_haritasi(r, dil):
    y = YAZI[dil]
    p = r.groupby(['arch', 'corruption']).rpc.mean().unstack().reindex(index=[a for a in SIRA if a in set(r.arch)], columns=BOZ)
    fig, a = plt.subplots(figsize=(8.6, 0.55 * len(p) + 1.4))
    im = a.imshow(p.values, cmap='Blues', vmin=np.nanmin(p.values) - 5, vmax=100, aspect='auto')
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.values[i, j]
            a.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=8.5, color='white' if v > 85 else 'black')
    a.set_xticks(range(len(BOZ)), [BAD[dil][b] for b in BOZ], rotation=30, ha='right', fontsize=9)
    a.set_yticks(range(len(p)), [AD[m] for m in p.index], fontsize=9)
    cb = fig.colorbar(im, ax=a, fraction=0.03)
    cb.set_label(y['rpc'], fontsize=9)
    fig.tight_layout()
    kaydet(fig, 'fig_heatmap_rpc', dil)

def sinif_kaybi(dil, arch='yolo11m', siddet=3):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    satir = []
    for d in glob.glob(str(KOK / 'results' / '*' / 'results' / 'pred' / f'{arch}_s*_s*')):
        gt = json.load(gzip.open(Path(d) / 'gt.json.gz', 'rt'))
        coco = COCO()
        coco.dataset = gt
        coco.createIndex()
        for b in ['clean'] + BOZ:
            f = Path(d) / (f'{b}_0.json.gz' if b == 'clean' else f'{b}_{siddet}.json.gz')
            if not f.exists():
                continue
            e = COCOeval(coco, coco.loadRes(json.load(gzip.open(f, 'rt'))), 'bbox')
            e.params.maxDets = [1, 10, 100]
            e.evaluate()
            e.accumulate()
            pr = e.eval['precision'][:, :, :, 0, 2]
            for k, cat in enumerate(e.params.catIds):
                v = pr[:, :, k]
                satir.append({'run': d, 'boz': b, 'cat': cat, 'ap': float(v[v > -1].mean()) * 100 if (v > -1).any() else np.nan})
    if not satir:
        return None
    s = pd.DataFrame(satir)
    s.to_csv(KOK / 'results' / 'tablolar' / f't4_sinif_{arch}.csv', index=False)
    return s

def kaydet(fig, ad, dil):
    d = KOK / 'makale' / ('sekiller' if dil == 'en' else 'sekiller_tr')
    d.mkdir(parents=True, exist_ok=True)
    for uz in ('png', 'svg'):
        fig.savefig(d / f'{ad}.{uz}', dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    plt.rcParams.update({'font.family': 'Calibri', 'font.size': 9})
    df = oku()
    oz = ozet(df)
    r = tablolar(df, oz)
    for dil in ('en', 'tr'):
        siddet_egrisi(df, dil)
        isi_haritasi(r, dil)
    sinif_kaybi('en')
if __name__ == '__main__':
    main()
