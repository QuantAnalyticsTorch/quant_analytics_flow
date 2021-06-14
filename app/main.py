from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

app = FastAPI()

import numpy as np
from scipy.stats import norm

from quant_analytics_flow.analytics import blackanalytics

@app.get("/analytics/blackscholes")
def blackscholes(fwd: float, strike: float, volatility: float, tau: float):
    z = blackanalytics.black(fwd, strike, tau, volatility)
    return { "price" : float(z) }

@app.get("/analytics/impliedvolatility")
def impliedvolatility(price: float, fwd: float, strike: float, tau: float):
    iv = blackanalytics.impliedvolatility(price, fwd, strike, tau)
    return { "impliedvolatility" : float(iv) }

@app.get("/analytics/blackscholes_tf")
def blackscholes_tf(fwd: float, strike: float, volatility: float, tau: float):
    forward = tf.Variable(fwd)
    strike = tf.Variable(strike)    
    vol = tf.Variable(volatility)        
    time = tf.Variable(tau)        
    r = tf.constant(0.0)                

    with tf.GradientTape() as tape:
        bsn = blackanalytics.black_tf(forward, strike, time, vol, r)

    dblack_dx = tape.gradient(bsn, [forward,vol])

    print(bsn)

    return { 
        "price" : float(bsn.numpy()),
        "delta" : float(dblack_dx[0].numpy()),
        "vega" : float(dblack_dx[1].numpy())
    }
