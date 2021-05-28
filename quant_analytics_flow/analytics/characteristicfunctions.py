import math
import tensorflow as tf


def test(A):
    """ Square root of positive semi-definite matrix
    
    .. math:: 
        Q^T \cdot Q = A
    
    using a singular value decomposition

    Args:
        A (tensor(shape=(N,N))): Symmetric 2-dimensional tensor

    Returns:
        Q (tensor(shape=(N,N))): Returns square root :math:`Q`

    """
    return 0.0