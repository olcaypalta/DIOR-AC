import glob
import gzip
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from analiz import BOZ, oku
KOK = Path(__file__).resolve().parents[1]

def holm(p):
    p = np.asarray(p)
    o = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    acc = 0
    for k, i in enumerate(o):
        acc = max(acc, (m - k) * p[i])
        adj[i] = min(1.0, acc)
    return adj

def eslestirilmis():
    df = oku()
    y = df[df.arch.isin(['yolo11m', 'yolo11s'])]
    olc = {}
    for (a, s), g in y.groupby(['arch', 'seed']):
        c = g[g.corruption == 'clean'].mAP.iloc[0]
        b = g[g.corruption != 'clean']
        d = {'clean': c, 'mPC': b.mAP.mean(), 'rPC': 100 * b.mAP.mean() / c}
        for k in BOZ:
            d[f'rPC_{k}'] = 100 * b[b.corruption == k].mAP.mean() / c
        olc[a, s] = d
    satir = []
    for k in olc['yolo11m', 0]:
        m = np.array([olc['yolo11m', s][k] for s in (0, 1, 2)])
        s_ = np.array([olc['yolo11s', s][k] for s in (0, 1, 2)])
        fark = m - s_
        t = stats.ttest_rel(m, s_)
        ga = stats.t.ppf(0.975, 2) * fark.std(ddof=1) / np.sqrt(3)
        w = stats.ttest_ind(m, s_, equal_var=False)
        satir.append({'olcut': k, 'm': m.mean(), 's': s_.mean(), 'fark': fark.mean(), 'fark_tohum': ' / '.join((f'{v:+.2f}' for v in fark)), 'ga_alt': fark.mean() - ga, 'ga_ust': fark.mean() + ga, 'p_eslestirilmis': t.pvalue, 'p_welch': w.pvalue})
    r = pd.DataFrame(satir)
    b = r.olcut.str.startswith('rPC_')
    r.loc[b, 'p_holm_eslestirilmis'] = holm(r.loc[b, 'p_eslestirilmis'].values)
    r.to_csv(KOK / 'results' / 'tablolar' / 't6_eslestirilmis.csv', index=False, float_format='%.4f')
    print(r.round(3).to_string())

def sinif_ozet(arch='yolo11m', siddet=3):
    s = pd.read_csv(KOK / 'results' / 'tablolar' / f't4_sinif_{arch}.csv')
    s['seed'] = s.run.str.extract('_s(\\d)$').astype(int)
    temiz = s[s.boz == 'clean'].set_index(['seed', 'cat']).ap
    satir = []
    for b in BOZ:
        g = s[s.boz == b].set_index(['seed', 'cat']).ap
        oran_tohum = (100 * g / temiz).unstack('cat')
        ort_oran = 100 * g.groupby('cat').mean() / temiz.groupby('cat').mean()
        for c in ort_oran.index:
            satir.append({'boz': b, 'cat': c, 'oran_ortalamalar': ort_oran[c], 'oran_tohum_ort': oran_tohum[c].mean(), 'oran_tohum_std': oran_tohum[c].std(ddof=1)})
    o = pd.DataFrame(satir)
    sayi = []
    for d in sorted(glob.glob(str(KOK / 'results' / '*' / 'results' / 'pred' / f'{arch}_s*_pred_s*'))):
        gt = json.load(gzip.open(Path(d) / 'gt.json.gz', 'rt'))
        sayi.append(pd.Series([a['category_id'] for a in gt['annotations']]).value_counts())
    sayi = pd.concat(sayi, axis=1)
    o['nesne_min'] = o.cat.map(sayi.min(axis=1))
    o['nesne_max'] = o.cat.map(sayi.max(axis=1))
    o.to_csv(KOK / 'results' / 'tablolar' / f't4b_sinif_ozet_{arch}.csv', index=False, float_format='%.2f')
    sn = o[o.boz == 'sensor_noise'].sort_values('oran_ortalamalar')
    print(sn[['cat', 'oran_ortalamalar', 'oran_tohum_std', 'nesne_min', 'nesne_max']].round(1).to_string())
    for esik in (5, 10):
        print(f'sensor_noise sev{siddet}: {(sn.oran_ortalamalar < esik).sum()} / 20 sınıf < %{esik}')
if __name__ == '__main__':
    eslestirilmis()
    sinif_ozet()
