#import torch
import numpy as np
from scipy.stats import norm
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def black(s : float, k : float, dt : float, v : float, r=0.0):
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

@tf.function(experimental_compile=True)
def black_tf(s : tf.Tensor, k : tf.Tensor, dt : tf.Tensor, v : tf.Tensor, r : tf.Tensor):
    n = tfd.Normal(loc=0., scale=1.)
    sdt = v * tf.sqrt(dt)
    d1 = (tf.math.log(s / k) + (r + v * v / 2) * dt) / sdt
    d2 = d1 - sdt
    return s * n.cdf(d1) - k * n.cdf(d2)