

from __future__ import division
import os
import time
import math
from common import *
from model.nets import *

class NNlearner():

    def __init__():
        pass

    def parsing_size(self):
        opt = self.opt

        # get train/val nums and batch_nums for every epoch
        # batch_nums_per_epoch = nums / batch_size
        train_nums = get_dataset_size_from_name(
            opt.train_dataset_dir)
        val_nums = get_dataset_size_from_name(opt.val_dataset_dir)

        train_batch_nums_per_epoch = get_batch_nums_per_epoch(
            train_nums, opt.batch_size)
        val_batch_nums_per_epoch = get_batch_nums_per_epoch(
            val_nums, 1)

        max_epochs = get_max_epoch(
            opt.max_iteration,
            train_batch_nums_per_epoch)
        # print self.train_nums,self.train_batch_nums_per_epoch,self.max_epochs

        # collect params needed.
        self.train_nums = train_nums
        self.val_nums = val_nums
        self.train_batch_nums_per_epoch = train_batch_nums_per_epoch
        self.val_batch_nums_per_epoch = val_batch_nums_per_epoch
        self.max_epochs = max_epochs

    def build_train_graph(self):
        '''
        Training/Validation Data Loader Part
        '''
        opt = self.opt

        with tf.name_scope("train_data_loader"):

            train_idx, train_sparse_dep, train_rgb, train_dense_dep = input_fn(
                opt.train_dataset_dir, opt.batch_size, self.train_nums, self.seed)
            val_idx, val_sparse_dep, val_rgb, val_dense_dep = input_fn(
                opt.val_dataset_dir, 1, self.val_nums, self.seed)

            # choose data from train or validation
            is_training = tf.placeholder(dtype=bool, shape=())
            index = tf.cond(is_training, lambda: train_idx, lambda: val_idx)
            sparse_dep = tf.cond(
                is_training,
                lambda: train_sparse_dep,
                lambda: val_sparse_dep)
            rgb = tf.cond(
                is_training,
                lambda: train_rgb,
                lambda: val_rgb)
            dense_dep = tf.cond(
                is_training,
                lambda: train_dense_dep,
                lambda: val_dense_dep)


        return