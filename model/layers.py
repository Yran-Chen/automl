import tensorflow as tf

def hidden_layer(
        input,
        hidden_unit,
        activation='relu',
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weight_decay=0.0001,
        name=""):

    with tf.variable_scope(name) as sc:

        in_unit = input.get_shape()

        w = tf.Variable(
        name = 'weights',
        shape = [in_unit,hidden_unit],
        initializer = weights_initializer,
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
        trainable = True
        )

        b = tf.Variable(
        name = 'bias',
        shape = [hidden_unit],
        initializer = weights_initializer,
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay),
        trainable = True
        )

        feature = tf.matmul(input, w) + b

        if activation == 'relu':
            feature = tf.nn.relu(feature)
        elif activation == 'sigmoid':
            feature = tf.nn.sigmoid(feature)
        elif activation == 'softmax':
            feature == tf.nn.softmax(feature)

    return feature
