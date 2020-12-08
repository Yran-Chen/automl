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

import pandas as pd
import numpy as np
import os

OP_DICT = ['zscore', 'cbrt', 'sigmoid', 'stdscaler','freq']

import tensorflow as tf
from tensorflow import keras
import numpy as np


# DEFINE NETWORK.
baseline_net = keras.Sequential([
        layers.Dense(450, activation='relu', input_shape=(400,)),
        layers.Dense(2, activation='softmax')
    ])
baseline_net.compile(optimizer=keras.optimizers.Adam(),
                         loss=keras.losses.SparseCategoricalCrossentropy(),
                         metrics=['accuracy'])
baseline_net.summary()

"""
Improved Network.
"""
imp_net = keras.Sequential([
        layers.Dense(450, activation='relu', input_shape=(400,)),
        layers.Dense(500, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
imp_net.compile(optimizer=keras.optimizers.Adam(),
                loss=tf.keras.losses.BinaryCrossentropy(),
                         metrics=['accuracy'])
imp_net.summary()

relu_net = keras.Sequential([
        layers.Dense(450, activation='softmax', input_shape=(400,)),
        layers.Dense(500, activation='softmax'),
        layers.Dense(1, activation='relu')
    ])

from sklearn.metrics import f1_score, recall_score, precision_score
from keras.callbacks import Callback

class Metrics(Callback):
  def __init__(self, val_data, val_label):
      super(Callback, self).__init__()
      self.val_data = val_data
      self.val_label = val_label

  def on_train_begin(self, logs={}):
    self.val_f1s = []

  def on_epoch_end(self, epoch, logs={}):
    val_predict=(np.asarray(self.model.predict_classes(self.val_data,batch_size=1))).round()
    val_targ = self.val_label
    _val_f1 = f1_score(val_targ, val_predict, average='binary')
    self.val_f1s.append(_val_f1)
    print("— val_f1: %f " % _val_f1)
    return

class LFE_learner():

    def __init__(self, param):
        self.name = param['name']
        self.train_name = param['train_name']
        self.data_dir = param['data_dir']
        self.eval_data_dir = param[ 'eval_data_dir']
        self.operator = param['operator']
        self.model = param['model']
        self.data_dir = os.path.join(self.data_dir, 'dataset', self.name)
        self.save_dir = os.path.join(self.data_dir, 'savefile', self.train_name)


        self.if_balanced = param['if_balanced']
        self.balance_percent = param['balance_percent']
        self.balance_method = param['balance_method']
        self.epoch = param['epoch']


    def train_forward(self, hyprprm=None):
        from keras.callbacks import TensorBoard
        from keras.callbacks import ModelCheckpoint
        import matplotlib.pyplot as plt

        for oprtr in self.operator:
            data_path = os.path.join(self.data_dir, oprtr, 'data.csv')
            data = self.load_data(data_path)

            print ('——'*40)
            print(oprtr)
            # print(data)

            x_train, x_test, y_train, y_test = self.split_data(data, test_size=0.2)

            print ( 'pos percent : {}'.format(y_train.sum() / len(y_train) )   )
            print ( 'Train on {} samples, Eval on {}.'.format(len(y_train),len(y_test)) )

            model = self.init_network()

            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(self.save_dir, oprtr, 'log'))

            checkpoint = ModelCheckpoint(filepath=os.path.join(self.save_dir, oprtr,
                                                               # "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
                                                               'best_val_acc-{epoch:02d}-{val_accuracy:.4f}.hdf5'
                                                               ),
                                         monitor='val_accuracy',
                                         mode='auto',
                                         save_best_only=False,
                                         period=5,
                                         )

            __metrics = Metrics(x_test, y_test)
            callback_lists = [tensorboard, checkpoint, __metrics]

            history = model.fit(x_train, y_train,
                                batch_size=4, epochs=self.epoch,
                                validation_data=(x_test, y_test),
                                verbose=0,
                                callbacks=callback_lists)

            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.plot(__metrics.val_f1s)
            plt.legend(['training', 'validation', 'val_f1'], loc='lower left')
            plt.savefig(r'{}\{}.png'.format(self.save_dir, oprtr))
            # plt.show()
            plt.close()

    def split_data(self, data, test_size=0.2):
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(data[:, 0:-1], data[:, -1],
                                                            test_size=test_size,
                                                            stratify=data[:, -1],
                                                            random_state=99)

        train_set = np.concatenate((x_train, y_train.reshape(-1, 1)), axis=1)

        if self.if_balanced:
            train_set = self.banlance_data(train_set, self.balance_percent, method = self.balance_method)

        np.random.shuffle(train_set)

        return train_set[:, 0:-1], x_test, train_set[:, -1], y_test

    def eval_forward(self,if_pos = False):
        from keras.models import load_model

        for oprtr in self.operator:
            data_path = os.path.join(self.eval_data_dir, oprtr, 'data.csv')
            data = self.load_data(data_path)

            x_test = data[:, 0:-1]
            y_test = data[:, -1]

            if if_pos:
                x_test = pd.DataFrame(x_test)
                y_test = pd.DataFrame(y_test)
                idx = y_test.loc[y_test.iloc[:, -1] == 1].index
                x_test = np.array(x_test.loc[idx])
                y_test = np.array(y_test.loc[idx])

            if len(y_test) != 0:
                print ( 'pos percent : {}'.format(y_test.sum() / len(y_test) )   )
            else:
                print('pos percent : 0.0. Stop eval.')
                continue

            model_path = self.load_model(oprtr)
            model = tf.keras.models.load_model(model_path)

            loss, accuracy = model.evaluate(x_test, y_test, batch_size=1,verbose=0)
            # print('\ntest loss', loss)
            print('accuracy: ', accuracy)

            y_pred = model.predict_classes(x_test)
            _val_f1 = f1_score(y_test, y_pred, average='binary')
            print ("F1 score: ",_val_f1)

            print ('Oprtr', oprtr)
            print ('——'*40)


    def eval(self,data,oprtr):
        from keras.models import load_model

        model_path = self.load_model(oprtr)
        model = tf.keras.models.load_model(model_path)

        y_pred = model.predict_classes(data)

        return y_pred


    def load_model(self,oprtr,key_monitor = 'acc'):
        best_model = None
        model_list = os.listdir(
            os.path.join(self.save_dir, oprtr)
        )
        for mdl in model_list:
            if mdl.startswith('best') and mdl.endswith('hdf5'):
                best_model = mdl
        return os.path.join(self.save_dir,oprtr,best_model)

    def val_model(self, model_path, oprtr, if_pos=False):
        from keras.models import load_model

        data_path = os.path.join(self.data_dir, oprtr, 'data.csv')
        data = self.load_data(data_path)

        x_test = data[:, 0:-1]
        y_test = data[:, -1]

        if if_pos:
            x_test = pd.DataFrame(x_test)
            y_test = pd.DataFrame(y_test)
            idx = y_test.loc[y_test.iloc[:, -1] == 1].index
            x_test = np.array(x_test.loc[idx])
            y_test = np.array(y_test.loc[idx])

        model = tf.keras.models.load_model(model_path)

        return model.evaluate(x_test, y_test, batch_size=4)

    def banlance_data(self, data, perc,method = 'downsample'):

        if method == 'downsample':

            df = pd.DataFrame(data)

            pos_data = df.loc[df.iloc[:, -1] == 1.0]
            neg_data = df.loc[df.iloc[:, -1] == 0.0]
            pos_num = len(pos_data)
            neg_num = len(neg_data)
            total_num = len(df)

            if (pos_num/total_num)<perc:
                downsample_num = int((perc * total_num - (perc) * neg_num) / (1 - perc))
                df = pd.concat([pos_data, neg_data.sample(n=downsample_num, replace=False, axis=0)], axis=0)

            else:
                downsample_num = int((perc * total_num - (perc) * pos_num) / (1 - perc))
                df = pd.concat([neg_data, pos_data.sample(n=downsample_num, replace=False, axis=0)], axis=0)

        elif method == 'upsample':

            df = pd.DataFrame(data)

            pos_data = df.loc[df.iloc[:, -1] == 1]
            pos_num = len(pos_data)
            total_num = len(df)

            oversample_num = int((perc * total_num - (1.0 + perc) * pos_num) / (1 - perc))
            df = pd.concat([df, pos_data.sample(n=oversample_num, replace=True, axis=0)], axis=0)

        return np.array(df)

    def init_network(self):
        return self.model

    def load_data(self, path):
        return np.array(pd.read_csv(path, header=None))

if __name__ == '__main__':

    train_param = {
        'name': 'test_beta',
        'data_dir': r"D:\!DTStack\Savefile_remote",
        'eval_data_dir': r'D:\!DTStack\Savefile_remote\dataset\test_beta_eval_00489',
        'train_name': 'demo_model_balanced',
        'operator': OP_DICT,
        'model': imp_net,
        'if_balanced': True,
        'balance_percent': 0.5005,
        'balance_method': 'downsample',
        'epoch': 300,
    }

    lqf = LFE_learner(train_param)
    # lqf.train_forward()
    lqf.eval_forward()
