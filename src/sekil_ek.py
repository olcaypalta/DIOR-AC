import glob
import gzip
import io
import json
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analiz import AD, BAD, BOZ, RENK, SIRA, kaydet, oku, ozet
from bozulma import uygula
from degerlendir import SINIFLAR
KOK = Path(__file__).resolve().parents[1]
SINIF_TR = ['uçak', 'havalimanı', 'beyzbol sahası', 'basketbol sahası', 'köprü', 'baca', 'baraj', 'otoyol dinlenme tesisi', 'otoyol gişesi', 'golf sahası', 'atletizm pisti', 'liman', 'üst geçit', 'gemi', 'stadyum', 'depolama tankı', 'tenis kortu', 'tren istasyonu', 'araç', 'rüzgâr türbini']
TEMIZ = {'en': 'Clean', 'tr': 'Temiz'}

def test_df(n, seed):
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(KOK / 'veri' / 'test-*.parquet')))])
    return df.sample(n=n, random_state=seed).reset_index(drop=True)

def goruntu(r):
    return np.array(Image.open(io.BytesIO(r.image['bytes'])).convert('RGB'))

def font(b):
    try:
        return ImageFont.truetype('calibri.ttf', b)
    except OSError:
        return ImageFont.load_default()

def ornekler(dil, df, kare=260):
    secim, gor = ([], set())
    for sinif in (0, 13, 18):
        for _, r in df.iterrows():
            if r.image_id not in gor and sinif in list(r.objects['category']) and (len(r.objects['category']) >= 3):
                secim.append(r)
                gor.add(r.image_id)
                break
    sutun = ['clean'] + BOZ
    ust = 30
    tuval = Image.new('RGB', (kare * len(sutun), ust + kare * len(secim)), 'white')
    d = ImageDraw.Draw(tuval)
    f = font(17)
    for j, ad in enumerate(sutun):
        t = TEMIZ[dil] if ad == 'clean' else BAD[dil][ad]
        d.text((j * kare + (kare - d.textlength(t, font=f)) / 2, 5), t, fill='black', font=f)
    for i, r in enumerate(secim):
        x = goruntu(r)
        for j, ad in enumerate(sutun):
            y = uygula(x, ad, 3, np.random.default_rng(7))
            tuval.paste(Image.fromarray(y).resize((kare - 4, kare - 4)), (j * kare + 2, ust + i * kare + 2))
    hedef = KOK / 'makale' / ('sekiller' if dil == 'en' else 'sekiller_tr')
    tuval.save(hedef / 'fig_examples.png')

def nitel(dil, seed=0, esik=0.3):
    pdir = Path(glob.glob(str(KOK / 'results' / '*' / 'results' / 'pred' / f'yolo11m_s{seed}_pred_s{seed}'))[0])
    gt = json.load(gzip.open(pdir / 'gt.json.gz', 'rt'))
    df = test_df(len(gt['images']), seed)
    sayi = pd.Series([a['image_id'] for a in gt['annotations']]).value_counts()
    assert all((len(df.iloc[i].objects['category']) == sayi.get(i, 0) for i in range(len(df)))), 'gt eşleşmiyor'
    kosul = [('clean', 0), ('haze', 3), ('stripe_noise', 2), ('sensor_noise', 2), ('line_dropout', 2)]
    adaylar = [i for i in sayi.index if 6 <= sayi[i] <= 25][:40]
    secim = [adaylar[k] for k in (0, 7, 15)]
    son = max(secim)
    kare, ust = (300, 30)
    tuval = Image.new('RGB', (kare * len(kosul), ust + kare * len(secim)), 'white')
    d = ImageDraw.Draw(tuval)
    f = font(17)
    for j, (ad, s) in enumerate(kosul):
        t = TEMIZ[dil] if ad == 'clean' else f'{BAD[dil][ad]} ({s})'
        d.text((j * kare + (kare - d.textlength(t, font=f)) / 2, 5), t, fill='black', font=f)
        tahmin = json.load(gzip.open(pdir / f'{ad}_{s}.json.gz', 'rt'))
        rng = np.random.default_rng(1000 * seed + 17 * s + sum(map(ord, ad)))
        for i in range(son + 1):
            img = uygula(goruntu(df.iloc[i]), ad, s, rng) if ad != 'clean' else goruntu(df.iloc[i])
            if i not in secim:
                continue
            im = Image.fromarray(img)
            dd = ImageDraw.Draw(im)
            for p in tahmin:
                if p['image_id'] == i and p['score'] >= esik:
                    x, y, w, h = p['bbox']
                    dd.rectangle([x, y, x + w, y + h], outline=(237, 125, 49), width=4)
            k = secim.index(i)
            tuval.paste(im.resize((kare - 4, kare - 4)), (j * kare + 2, ust + k * kare + 2))
    hedef = KOK / 'makale' / ('sekiller' if dil == 'en' else 'sekiller_tr')
    tuval.save(hedef / 'fig_qualitative.png')

def sinif(dil):
    s = pd.read_csv(KOK / 'results' / 'tablolar' / 't4_sinif_yolo11m.csv')
    m = s.groupby(['boz', 'cat']).ap.mean().unstack()
    rel = 100 * m.loc[BOZ] / m.loc['clean']
    sira = m.loc['clean'].sort_values(ascending=False).index
    rel = rel[sira]
    adlar = [(SINIFLAR if dil == 'en' else SINIF_TR)[c] for c in sira]
    fig, a = plt.subplots(figsize=(10, 4.2))
    im = a.imshow(rel.values, cmap='Blues', vmin=0, vmax=100, aspect='auto')
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            v = rel.values[i, j]
            a.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=7, color='white' if v > 70 else 'black')
    a.set_yticks(range(len(BOZ)), [BAD[dil][b] for b in BOZ], fontsize=9)
    a.set_xticks(range(len(adlar)), adlar, rotation=45, ha='right', fontsize=8.5)
    cb = fig.colorbar(im, ax=a, fraction=0.025)
    cb.set_label('AP (sev. 3) / AP (clean), %' if dil == 'en' else 'AP (şiddet 3) / AP (temiz), %', fontsize=9)
    fig.tight_layout()
    kaydet(fig, 'fig_class', dil)
    return rel

def aileler(dil, oz):
    g = oz.groupby('arch').agg(c=('clean', 'mean'), cs=('clean', 'std'), r=('rPC', 'mean'), rs=('rPC', 'std'))
    g = g.reindex([m for m in SIRA if m in g.index])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    x = np.arange(len(g))
    for k, (col, sd, lab) in enumerate([('c', 'cs', 'Clean mAP (%)' if dil == 'en' else 'Temiz mAP (%)'), ('r', 'rs', 'rPC (%)')]):
        ax[k].bar(x, g[col], yerr=g[sd], color=[RENK[m] for m in g.index], capsize=3, width=0.65)
        ax[k].set_xticks(x, [AD[m] for m in g.index], rotation=20, ha='right', fontsize=9)
        ax[k].set_ylabel(lab, fontsize=9)
        ax[k].grid(axis='y', alpha=0.3, lw=0.5)
        for i, v in enumerate(g[col]):
            ax[k].text(i, v + g[sd].iloc[i] + 1.2, f'{v:.1f}', ha='center', fontsize=8)
    fig.tight_layout()
    kaydet(fig, 'fig_families', dil)

def main():
    plt.rcParams.update({'font.family': 'Calibri', 'font.size': 9})
    oz = ozet(oku())
    df0 = test_df(500, 0)
    for dil in ('en', 'tr'):
        ornekler(dil, df0)
        nitel(dil)
        sinif(dil)
        aileler(dil, oz)
    print('ok')
if __name__ == '__main__':
    main()
