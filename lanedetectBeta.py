import tensorflow as tf

def autoencoder(x):
    layer_1 = tf.layers.dense(x, n_hidden_1)
    layer_2 = tf.layers.dense(x, n_hidden_1)
    layer_3 = tf.layers.dense(x, n_hidden_1)
init = tf.initialize_all_variables()