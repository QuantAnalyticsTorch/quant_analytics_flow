import tensorflow as tf

from quant_analytics_flow.analytics import matrixanalytics
from quant_analytics_flow.analytics import constants

import tensorflow as tf

def test_square_root_symmetric_matrix():
    
    fm = tf.constant([[1.,0.5],[0.5,1.]], dtype=tf.float64)

    tms = matrixanalytics.square_root_symmetric_matrix(fm)

    assert tf.abs(tms[0][0] - 0.96592583)  < constants.EPSILON

if __name__ == '__main__':
    
    test_square_root_symmetric_matrix()