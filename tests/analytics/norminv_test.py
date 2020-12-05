from quant_analytics_flow.analytics import norminv
from quant_analytics_flow.analytics import constants

import tensorflow as tf

def test_norminv():
    
    x = tf.constant(0.5, dtype=tf.float64)

    y = norminv.norminv(x)

    assert y < constants.EPSILON

if __name__ == '__main__':
    
    test_norminv()