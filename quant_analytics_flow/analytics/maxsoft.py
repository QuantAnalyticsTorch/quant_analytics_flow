import tensorflow as tf
from quant_analytics_flow.analytics import constants

def phi_smooth(x,y):
    return (x + y + constants.DELTA + (x - y) ** 2 / constants.DELTA / 4.)/2.

def max_if(x,y):
    return tf.where(tf.abs(x-y)>constants.BOUNDARY,tf.maximum(x,y),phi_smooth(x,y))

def hyperbolic(x):
      """ Using the hyperbolic function 
    
      .. _target hyperbolic_function:

      .. math::

        f(x) = \\frac{1}{2} \\left(x + \sqrt{1 + x^2} \\right)

      Args:
          x (tensor(shape=(...))): M-dimensional tensor

      Returns:
          y (tensor(shape=(...))): Hyperbolic function

      .. jupyter-execute::          

          import matplotlib.pyplot as plt
          import quant_analytics_flow.analytics.maxsoft as maxsoft
          import tensorflow as tf
          x = tf.cast(tf.linspace(-1.0,1.0,11), dtype=tf.float64)
          y = maxsoft.hyperbolic(x)
          plt.plot(x.numpy(),y.numpy())
          
      """
      
      return (x + tf.sqrt(1. + x*x))/2.

def soft_max_hyperbolic(x,eps=constants.EPSILON):
      """ Using the :ref:`hyperbolic function <target hyperbolic_function>` to approximate :math:`\max(x,0)`
    
      .. _target soft_max_hyperbolic:

      .. math::

          g_(x) = f(x/\\epsilon)\cdot \\epsilon

      Args:
          x (tensor(shape=(...))): M-dimensional tensor
          eps (float64): scaling parameter

      Returns:
          y (tensor(shape=(...))): Hyperbolic function

      .. jupyter-execute::          

          import matplotlib.pyplot as plt
          import quant_analytics_flow.analytics.maxsoft as maxsoft
          import tensorflow as tf
          x = tf.cast(tf.linspace(-1.0,1.0,21), dtype=tf.float64)
          y = maxsoft.hyperbolic(x)
          z = maxsoft.soft_max_hyperbolic(x)                  
          w = maxsoft.soft_max_hyperbolic(x, tf.constant(0.1, dtype=tf.float64))          
          plt.plot(x.numpy(),y.numpy())
          plt.plot(x.numpy(),z.numpy())
          plt.plot(x.numpy(),w.numpy())          


      """
      return hyperbolic(x/eps)*eps