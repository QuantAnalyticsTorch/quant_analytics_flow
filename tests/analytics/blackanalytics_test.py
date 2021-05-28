import tensorflow as tf

from quant_analytics_flow.analytics import blackanalytics
from quant_analytics_flow.analytics import constants

def test_black():
    
    forward = 100.0;
    strike = 100.0;
    vol = 0.2;
    time = 1.0;
    r = 0.0;

    bsn = blackanalytics.black(forward, strike, time, vol)
    bsv = blackanalytics.black_vega(forward, strike, time, vol)
    bsg = blackanalytics.black_gamma(forward, strike, time, vol)

    assert tf.abs(bsn - 7.965567455405804)  < constants.EPSILON
    assert tf.abs(bsg - 0.01984762737385059)  < constants.EPSILON
    assert tf.abs(bsv - 39.69525474770118)  < constants.EPSILON

def test_implied_black():
    
    forward = 100.0;
    strike = 100.0;
    vol = 0.3;
    time = 1.0;
    r = 0.0;

    bsn = blackanalytics.black(forward, strike, time, vol)
    iv = blackanalytics.impliedvolatility(bsn, forward, strike, time)

    assert tf.abs(iv - vol)  < constants.EPSILON

if __name__ == '__main__':
    
    test_black()