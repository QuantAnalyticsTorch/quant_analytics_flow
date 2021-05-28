#import torch
import numpy as np
from scipy.stats import norm

def black(s, k, dt, v, r=0.0):
    std = v * np.sqrt(dt);
    d1 = np.log(s / k) / std + 0.5 * std;
    return s * norm.cdf(d1) - k * norm.cdf(d1 - std);

def black_vega(s, k, dt, v, r=0.0):
    std = v * np.sqrt(dt);
    d1 = np.log(s / k) / std + 0.5 * std;
    return s * norm.pdf(d1) * np.sqrt(dt);

def black_gamma(s, k, dt, v, r=0.0):
    std = v * np.sqrt(dt);
    d1 = np.log(s / k) / std + 0.5 * std;
    return norm.pdf(d1)/s/std;

def impliedvolatility(p, s, k, dt, feps = 1e-8, veps = 1e-8):
    v = 0.2
    vp = 0
    pt = black(s, k, dt, v)

    while((np.abs(p-pt)>feps) and (np.abs(v-vp)>veps)):
        vp = v
        vega = black_vega(s, k, dt, v)
        v = v - (pt-p)/vega;
        pt = black(s, k, dt, v)

    return v