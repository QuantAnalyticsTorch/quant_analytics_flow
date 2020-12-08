import tensorflow as tf

from quant_analytics_flow.analytics import maxsoft
from quant_analytics_flow.analytics import constants

def test_hyperbolic():
    
    x = tf.constant(0.5, dtype=tf.float64)

    y = maxsoft.hyperbolic(x)

    assert tf.abs(y-0.8090169943749475) < constants.EPSILON

def test_soft_max_hyperbolic():
    
    x = tf.constant(-0.5, dtype=tf.float64)

    y = maxsoft.soft_max_hyperbolic(x)

    assert tf.abs(y-3.725290298461914e-17) < constants.EPSILON

def test_max_if():
    
    x = tf.constant(0.00000002, dtype=tf.float64)
    y = tf.constant(0.00000001, dtype=tf.float64)

    z = maxsoft.max_if(x,y)

    assert tf.abs(z-2.030330085889911e-08) < constants.EPSILON

if __name__ == '__main__':
    
    test_max_if()