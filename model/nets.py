import  tensorflow as tf
from model.layers import *


class NetHandler():

    def __init__(
            self,
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weight_decay=0.0001,

    ):
        self.weights_initializer = weights_initializer
        self.weight_decay = weight_decay

    def multilayer_perceptron(self,input_x, hidden_unit_num = 450):

        with tf.variable_scope('multilayer_perceptron') as sc:

            h1_output = hidden_layer(
                input_x,
                hidden_unit = hidden_unit_num,
                activation='relu',
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weight_decay=0.0,
                name="hidden_1"
            )

            h2_output = hidden_layer(
                h1_output,
                hidden_unit=2,
                activation='softmax',
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weight_decay=0.0,
                name="hidden_1"

            )

        return h2_output
