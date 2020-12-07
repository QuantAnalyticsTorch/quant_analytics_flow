import tensorflow as tf

def square_root_symmetric_matrix(A):
    w, v = tf.linalg.eigh(A)
    return tf.matmul(tf.matmul(v, tf.linalg.diag(tf.sqrt(w))),v)