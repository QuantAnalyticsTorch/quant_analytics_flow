import tensorflow as tf
from quant_analytics_flow.analytics import constants

def phi_smooth(x,y):
    return (x + y + constants.DELTA + (x - y) ** 2 / constants.DELTA / 4.)/2.

def max_if(x,y):
    return tf.where(tf.abs(x-y)>constants.BOUNDARY,tf.maximum(x,y),phi_smooth(x,y))

def hyperbolic(x):
  return (x + tf.sqrt(1. + x*x))/2.

def soft_max_hypterbolic(x,eps=constants.EPSILON):
  return hyperbolic(x/eps)*eps