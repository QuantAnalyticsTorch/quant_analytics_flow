from quant_analytics_flow.calculators.univariatebrownianbridge import UnivariateBrownianBridge
from quant_analytics_flow.analytics.norminv import norminv
from quant_analytics_flow.analytics import constants

import tensorflow as tf

def test_brownian_path():

    dim =  2
    states = 1
    number =  7

    brownian = UnivariateBrownianBridge(dim)

    # Draw Sobol numbers
    x = tf.math.sobol_sample(dim,number,dtype=tf.dtypes.float64)
    x = tf.transpose(x)
    y = tf.reshape(x, shape=(dim,states,number))
    z = norminv(y)

    w = brownian.path(z, False)

    assert tf.abs(w[0][0][0]) < constants.EPSILON
    assert tf.abs(w[1][0][1]) < constants.EPSILON

def test_brownian_path_increment():
    
    dim =  2
    states = 1
    number =  7

    brownian = UnivariateBrownianBridge(dim)

    # Draw Sobol numbers
    x = tf.math.sobol_sample(dim,number,dtype=tf.dtypes.float64)
    x = tf.transpose(x)
    y = tf.reshape(x, shape=(dim,states,number))
    z = norminv(y)

    w = brownian.path(z, True)

    assert tf.abs(w[0][0][0]) < constants.EPSILON
    assert tf.abs(w[1][0][1] - tf.constant(9.53872552e-01, dtype=tf.float64)) < constants.EPSILON


if __name__ == '__main__':
    
    test_brownian_path_increment()