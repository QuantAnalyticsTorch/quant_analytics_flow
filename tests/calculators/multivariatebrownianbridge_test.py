import tensorflow as tf

from quant_analytics_flow.analytics import constants

from quant_analytics_flow.calculators.multivariatebrownianbridge import MultivariateBrownianBridge

def test_multivariate_brownian_path():
    
    fm = tf.constant([[1.,0.5],[0.5,1.]], dtype=tf.float64)

    fwdCov = tf.TensorArray(dtype=tf.float64, size = 3)
    fwdCov = fwdCov.write(0,fm)
    fwdCov = fwdCov.write(1,fm)
    fwdCov = fwdCov.write(2,fm)        

    multivariate_bridge = MultivariateBrownianBridge(fwdCov.stack())

    w = multivariate_bridge.path(4)

    assert tf.abs(w[0][0][1] + 0.39913046) < constants.EPSILON
    assert tf.abs(w[1][1][1] - 0.64601085) < constants.EPSILON


if __name__ == '__main__':
    
    test_multivariate_brownian_path()