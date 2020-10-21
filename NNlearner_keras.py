# python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/10/20 15:14
# @Author  : Yran CHan
# @Site    : 
# @File    : NNlearner_keras.py
# @Software: PyCharm



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)



if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape([x_train.shape[0], -1])
    x_test = x_test.reshape([x_test.shape[0], -1])
    print(x_train.shape, ' ', y_train.shape)
    print(x_test.shape, ' ', y_test.shape)

    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()


    history = model.fit(x_train, y_train, batch_size=256, epochs=100, validation_split=0.3, verbose=0)

    import matplotlib.pyplot as plt

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()

    result = model.evaluate(x_test, y_test)