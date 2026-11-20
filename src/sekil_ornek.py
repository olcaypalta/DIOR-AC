import glob
import io
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from bozulma import BOZULMALAR, uygula
ETIKET = {'clean': 'Clean', 'haze': 'Haze', 'thin_cloud': 'Thin cloud', 'stripe_noise': 'Stripe noise', 'jpeg2000': 'JPEG2000', 'mtf_blur': 'MTF blur', 'sensor_noise': 'Sensor noise', 'low_sun': 'Low sun', 'line_dropout': 'Line dropout'}

def main(kok='/workspace/dior', cikti='/workspace/results/fig_examples.jpg', siddet=3, kare=260):
    df = pd.read_parquet(sorted(glob.glob(f'{kok}/data/test-*.parquet'))[0])
    secim, gorulen = ([], set())
    for sinif in (0, 13, 18):
        for _, r in df.iterrows():
            if r.image_id not in gorulen and sinif in list(r.objects['category']) and (len(r.objects['category']) >= 3):
                secim.append(r)
                gorulen.add(r.image_id)
                break
    sutun = ['clean'] + list(BOZULMALAR)
    ust = 26
    tuval = Image.new('RGB', (kare * len(sutun), ust + kare * len(secim)), 'white')
    d = ImageDraw.Draw(tuval)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 15)
    except OSError:
        font = ImageFont.load_default()
    for j, ad in enumerate(sutun):
        w = d.textlength(ETIKET[ad], font=font)
        d.text((j * kare + (kare - w) / 2, 4), ETIKET[ad], fill='black', font=font)
    for i, r in enumerate(secim):
        x = np.array(Image.open(io.BytesIO(r.image['bytes'])).convert('RGB'))
        for j, ad in enumerate(sutun):
            y = uygula(x, ad, siddet, np.random.default_rng(7))
            tuval.paste(Image.fromarray(y).resize((kare - 4, kare - 4)), (j * kare + 2, ust + i * kare + 2))
    tuval.save(cikti, quality=88)
    print(cikti, tuval.size)
if __name__ == '__main__':
    main(*sys.argv[1:])
