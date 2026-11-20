import glob
from pathlib import Path
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analiz import AD, BAD, RENK, kaydet, oku
KOK = Path(__file__).resolve().parents[1]
T = KOK / 'results' / 'tablolar'

def ms(x):
    return f'{x.mean():.2f} ± {x.std(ddof=1):.2f}'

def taban():
    df = oku()
    y = df[df.arch.isin(['yolo11m', 'yolo11s'])]
    return y

def onarim_tablo(y):
    on = pd.concat([pd.read_csv(f) for f in glob.glob(str(KOK / 'results' / 'onarim' / '*_onarim.csv'))])
    tm = pd.concat([pd.read_csv(f) for f in glob.glob(str(KOK / 'results' / 'onarim' / '*_onarim_temiz.csv'))])
    for d in (on, tm):
        d['arch'] = d.model.str.replace('_s\\d+$', '', regex=True)
    temiz = y[y.corruption == 'clean'][['arch', 'seed', 'mAP']].rename(columns={'mAP': 'clean'})
    boz = y[y.corruption != 'clean'][['arch', 'seed', 'corruption', 'severity', 'mAP']].rename(columns={'mAP': 'degraded'})
    k = on.rename(columns={'mAP': 'repaired'}).merge(boz, on=['arch', 'seed', 'corruption', 'severity']).merge(temiz, on=['arch', 'seed'])
    k['R'] = (k.repaired - k.degraded) / (k.clean - k.degraded)
    satir = []
    for (a, c, s), g in k.groupby(['arch', 'corruption', 'severity']):
        satir.append({'arch': a, 'corruption': c, 'severity': s, 'repair': g.repair.iloc[0], 'clean': ms(g.clean), 'degraded': ms(g.degraded), 'repaired': ms(g.repaired), 'R': ms(g.R)})
    t = pd.DataFrame(satir)
    t.to_csv(T / 't9_onarim.csv', index=False)
    tmc = tm.merge(temiz, on=['arch', 'seed'])
    tmc['fark'] = tmc.mAP - tmc.clean
    t2 = tmc.groupby(['arch', 'repair']).agg(clean_repaired=('mAP', ms), change=('fark', ms)).reset_index()
    t2.to_csv(T / 't9b_onarim_temiz.csv', index=False)
    print(t.to_string(), '\n', t2.to_string())
    return (k, tmc)

def ek_tablo(y):
    ek = pd.concat([pd.read_csv(f) for f in glob.glob(str(KOK / 'results' / 'onarim' / '*_ek.csv'))])
    ek['arch'] = ek.model.str.replace('_s\\d+$', '', regex=True)
    temiz = y[y.corruption == 'clean'][['arch', 'seed', 'mAP']].rename(columns={'mAP': 'clean'})
    ek = ek.merge(temiz, on=['arch', 'seed'])
    ek['rel'] = 100 * ek.mAP / ek.clean
    t = ek.groupby(['arch', 'corruption', 'param']).agg(mAP=('mAP', ms), rel=('rel', ms)).reset_index()
    t.to_csv(T / 't10_ek.csv', index=False)
    print(t.to_string())
    return ek

def sekil_onarim(k, tmc, dil):
    y = {'en': {'deg': 'degraded', 'rep': 'repaired', 'cl': 'clean', 'map': 'mAP (%)', 'sev': 'Severity', 'rp': {'satir_doldur': 'row interpolation', 'serit_gider': 'column normalisation', 'gurultu_gider': 'NL-means'}}, 'tr': {'deg': 'bozuk', 'rep': 'onarılmış', 'cl': 'temiz', 'map': 'mAP (%)', 'sev': 'Şiddet', 'rp': {'satir_doldur': 'satır enterpolasyonu', 'serit_gider': 'sütun normalizasyonu', 'gurultu_gider': 'NL-means'}}}[dil]
    sira = ['line_dropout', 'stripe_noise', 'sensor_noise']
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for i, c in enumerate(sira):
        a = ax[i]
        for arch, ls in (('yolo11m', '-'), ('yolo11s', '--')):
            g = k[(k.corruption == c) & (k.arch == arch)].groupby('severity')
            a.plot([1, 2, 3], g.degraded.mean(), ls, marker='o', ms=3.5, color='#A5A5A5', label=f'{AD[arch]} {y['deg']}')
            a.plot([1, 2, 3], g.repaired.mean(), ls, marker='o', ms=3.5, color=RENK[arch], label=f'{AD[arch]} {y['rep']}')
            cl = k[k.arch == arch].clean.mean()
            a.axhline(cl, color=RENK[arch], lw=0.8, ls=':')
        a.set_title(f'{BAD[dil][c]} + {y['rp'][k[k.corruption == c].repair.iloc[0]]}', fontsize=9.5)
        a.set_xticks([1, 2, 3])
        a.set_xlabel(y['sev'], fontsize=9)
        a.grid(alpha=0.3, lw=0.5)
    ax[0].set_ylabel(y['map'], fontsize=9)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    kaydet(fig, 'fig_repair', dil)

def sekil_kalite(y, dil):
    q = pd.read_csv(T / 't5_kalite.csv')
    temiz = y[y.corruption == 'clean'].groupby('arch').mAP.mean()
    m = y[(y.corruption != 'clean') & (y.arch == 'yolo11m')].groupby(['corruption', 'severity']).mAP.mean().reset_index()
    m['rel'] = 100 * m.mAP / temiz['yolo11m']
    m = m.merge(q, on=['corruption', 'severity'])
    renk = {'haze': '#4472C4', 'thin_cloud': '#9DC3E6', 'mtf_blur': '#70AD47', 'stripe_noise': '#ED7D31', 'sensor_noise': '#C00000', 'low_sun': '#FFC000', 'jpeg2000': '#7F7F7F', 'line_dropout': '#7030A0'}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for c, g in m.groupby('corruption'):
        ax[0].plot(g.psnr, g.rel, '-o', ms=4, color=renk[c], label=BAD[dil][c])
        ax[1].plot(g.ssim, g.rel, '-o', ms=4, color=renk[c])
    ax[0].set_xlabel('PSNR (dB)', fontsize=9)
    ax[1].set_xlabel('SSIM', fontsize=9)
    ax[0].set_ylabel('mAP / clean mAP (%)' if dil == 'en' else 'mAP / temiz mAP (%)', fontsize=9)
    for a in ax:
        a.grid(alpha=0.3, lw=0.5)
    fig.legend(*ax[0].get_legend_handles_labels(), loc='center right', fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    kaydet(fig, 'fig_quality', dil)

def main():
    plt.rcParams.update({'font.family': 'Calibri', 'font.size': 9})
    y = taban()
    k, tmc = onarim_tablo(y)
    ek_tablo(y)
    for dil in ('en', 'tr'):
        sekil_onarim(k, tmc, dil)
        sekil_kalite(y, dil)
if __name__ == '__main__':
    main()
