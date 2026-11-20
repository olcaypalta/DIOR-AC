import cv2
import numpy as np

def _smooth_noise(h, w, scale, rng):
    gh, gw = (max(2, h // scale), max(2, w // scale))
    g = rng.random((gh, gw)).astype(np.float32)
    g = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (0, 0), scale / 4)
    g -= g.min()
    return g / (g.max() + 1e-06)

def haze(x, s, rng):
    t = [0.75, 0.55, 0.4][s - 1]
    A = 0.85 + 0.1 * rng.random()
    J = x.astype(np.float32) / 255
    return np.clip((J * t + A * (1 - t)) * 255, 0, 255).astype(np.uint8)

def thin_cloud(x, s, rng):
    h, w = x.shape[:2]
    a = _smooth_noise(h, w, 96, rng) * [0.35, 0.55, 0.75][s - 1]
    J = x.astype(np.float32) / 255
    out = J * (1 - a[..., None]) + 0.95 * a[..., None]
    return np.clip(out * 255, 0, 255).astype(np.uint8)

def stripe_noise(x, s, rng):
    h, w = x.shape[:2]
    sd = [0.04, 0.08, 0.14][s - 1]
    gain = 1 + rng.normal(0, sd, size=(1, w, 1))
    off = rng.normal(0, sd * 40, size=(1, w, 1))
    return np.clip(x.astype(np.float32) * gain + off, 0, 255).astype(np.uint8)

def jpeg2000(x, s, rng):
    q = [100, 40, 15][s - 1]
    ok, buf = cv2.imencode('.jp2', cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, q])
    if not ok:
        raise RuntimeError('JPEG2000 kodlanamadı')
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def mtf_blur(x, s, rng):
    sig = [1.0, 2.0, 3.0][s - 1]
    return cv2.GaussianBlur(x, (0, 0), sig)

def sensor_noise(x, s, rng):
    peak = [60, 25, 10][s - 1]
    J = x.astype(np.float32) / 255
    n = rng.poisson(J * peak) / peak + rng.normal(0, [0.01, 0.02, 0.04][s - 1], J.shape)
    return np.clip(n * 255, 0, 255).astype(np.uint8)

def low_sun(x, s, rng):
    g = [1.6, 2.2, 3.0][s - 1]
    J = (x.astype(np.float32) / 255) ** g
    J = J * [0.8, 0.65, 0.5][s - 1]
    return np.clip(J * 255, 0, 255).astype(np.uint8)

def line_dropout(x, s, rng):
    h = x.shape[0]
    p = [0.03, 0.08, 0.15][s - 1]
    out = x.copy()
    rows = rng.random(h) < p
    out[rows] = 0
    return out
BOZULMALAR = {f.__name__: f for f in (haze, thin_cloud, stripe_noise, jpeg2000, mtf_blur, sensor_noise, low_sun, line_dropout)}

def uygula(x, ad, s, rng):
    return x if ad == 'clean' else BOZULMALAR[ad](x, s, rng)
