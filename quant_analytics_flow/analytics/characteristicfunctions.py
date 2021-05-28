import math

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
    w, v = tf.linalg.eigh(A)
    return tf.matmul(tf.matmul(v, tf.linalg.diag(tf.sqrt(w))),v)