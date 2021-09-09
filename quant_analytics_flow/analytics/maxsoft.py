import tensorflow as tf
from quant_analytics_flow.analytics import constants

def hyperbolic(x : tf.Tensor) -> tf.Tensor:
      """ Using the hyperbolic function 
    
      .. _target hyperbolic:

      .. math::

        f(x) = \\frac{1}{2} \\left(x + \sqrt{1 + x^2} \\right)

      Args:
          x (tf.Tensor): M-dimensional tensor

      Returns:
          y (tf.Tensor): Hyperbolic function

      .. jupyter-execute::          

          import matplotlib.pyplot as plt
          import quant_analytics_flow.analytics.maxsoft as maxsoft
          import tensorflow as tf
          x = tf.cast(tf.linspace(-1.0,1.0,31), dtype=tf.float64)
          y = maxsoft.hyperbolic(x)
          plt.plot(x.numpy(),y.numpy())
          
      """
      
      return (x + tf.sqrt(1. + x*x))/2.

def hyperbolic_prime(x : tf.Tensor) -> tf.Tensor:
      """ Using the derivative of the :hoverxref:`hyperbolic function <target hyperbolic>`
    
      .. _target hyperbolic_prime:

      .. math::

        f(x) = \\frac{1}{2}\\left(1 + \\frac{x}{\sqrt{1 + x^2}} \\right)

      Args:
          x (tf.Tensor): M-dimensional tensor

      Returns:
          y (tf.Tensor): Derivative of hyperbolic function

      .. jupyter-execute::          

          import matplotlib.pyplot as plt
          import quant_analytics_flow.analytics.maxsoft as maxsoft
          import tensorflow as tf
          x = tf.cast(tf.linspace(-1.0,1.0,31), dtype=tf.float64)
          y = maxsoft.hyperbolic_prime(x)
          plt.plot(x.numpy(),y.numpy())
          
      """
      
      return (1 + x/tf.sqrt(1. + x*x))/2.


def soft_max_hyperbolic(x,eps=constants.EPSILON):
      """ Using the :ref:`hyperbolic function <target hyperbolic>` to approximate :math:`\max(x,0)`
    
      .. _target soft_max_hyperbolic:

      .. math::

          g(x) = f(x/\\epsilon)\cdot \\epsilon

      Args:
          x (tensor(shape=(...))): M-dimensional tensor.
          eps (float64): scaling parameter.

      Returns:
          y (tensor(shape=(...))): Scaled hyperbolic function

      .. jupyter-execute::          

          import matplotlib.pyplot as plt
          import quant_analytics_flow.analytics.maxsoft as maxsoft
          import tensorflow as tf
          x = tf.cast(tf.linspace(-1.0,1.0,31), dtype=tf.float64)
          y = maxsoft.hyperbolic(x)
          z = maxsoft.soft_max_hyperbolic(x)                  
          w = maxsoft.soft_max_hyperbolic(x, tf.constant(0.1, dtype=tf.float64))          
          plt.plot(x.numpy(),y.numpy())
          plt.plot(x.numpy(),z.numpy())
          plt.plot(x.numpy(),w.numpy())          


      """
      return hyperbolic(x/eps)*eps

def soft_heavy_side_hyperbolic(x,eps=constants.EPSILON):
      """ Using the derivative of the :ref:`hyperbolic function <target hyperbolic_prime>` to approximate :math:`1_{(x>0)}`
    
      .. _target soft_heavy_side_hyperbolic:

      .. math::

          g(x) = f(x/\\epsilon)

      Args:
          x (tensor(shape=(...))): M-dimensional tensor
          eps (float64): scaling parameter

      Returns:
          y (tensor(shape=(...))): Scaled derivative of hyperbolic function

      .. jupyter-execute::          

          import matplotlib.pyplot as plt
          import quant_analytics_flow.analytics.maxsoft as maxsoft
          import tensorflow as tf
          x = tf.cast(tf.linspace(-1.0,1.0,31), dtype=tf.float64)
          y = maxsoft.hyperbolic_prime(x)
          z = maxsoft.soft_heavy_side_hyperbolic(x)                  
          w = maxsoft.soft_heavy_side_hyperbolic(x, tf.constant(0.1, dtype=tf.float64))          
          plt.plot(x.numpy(),y.numpy())
          plt.plot(x.numpy(),z.numpy())
          plt.plot(x.numpy(),w.numpy())          


      """
      return hyperbolic_prime(x/eps)