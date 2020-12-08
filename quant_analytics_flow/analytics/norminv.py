import tensorflow as tf

SQRT_2 = tf.sqrt(tf.constant(2.,dtype=tf.float64))

def norminv(x):
    """ Using the inverse error function :math:`\\text{erfinv}(x)` computes the inverse normal function
    
    .. _target norminv:

    .. math::

        \\text{norminv}(x) = \sqrt{2} \cdot \\text{erfinv}\\left( 2x - 1 \\right)

    Args:
        x (tensor(shape=(...))): M-dimensional tensor

    Returns:
        y (tensor(shape=(...))): Hyperbolic function

    .. jupyter-execute::          

        import matplotlib.pyplot as plt
        import quant_analytics_flow.analytics.norminv as norminv
        import tensorflow as tf
        x = tf.cast(tf.linspace(0.01,0.99,99), dtype=tf.float64)
        y = norminv.norminv(x)
        plt.plot(x.numpy(),y.numpy())
          
    """
    
    return SQRT_2*tf.math.erfinv(2*(x-0.5))