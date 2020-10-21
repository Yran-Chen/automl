# python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/10/20 14:28
# @Author  : Yran CHan
# @Site    : 
# @File    : loss.py
# @Software: PyCharm


import tensorflow as tf


def euclidean_loss(pre, gt, has_ignore_label=False, ignore_label=0):
    """ This func computes the euclidean distance between two tensorflow variables.
    loss = \frac{\sum_{i}^{N} (pre_{i} - gt_{i})^2 } {2xN}

    :param pre: the prediction results.
    :param gt: the groud truth
    :param has_ignore_label:  if True, consider ignore_label, set gt == ignore_label pixels' diff = 0
    :param ignore_label: the ignore_label value
    :return: euclidean loss result normed by batch_size
    """
    diff = tf.subtract(pre, gt)
    if has_ignore_label:
        ignore_label_constant = tf.constant(
            ignore_label / 255., dtype=tf.float32)
        where = tf.not_equal(gt, ignore_label_constant)
        indices = tf.where(where)
        diff = tf.gather_nd(diff, indices)
    else:
        diff = diff

    loss = tf.reduce_sum(tf.nn.l2_loss(diff))

    # norm loss by batch_size as N
    batch_size = tf.cast(tf.shape(pre)[0], tf.float32)
    loss = tf.div(loss, batch_size)

    return loss


def compute_rmse(pre, gt, ignore_zero):
    """
    Compute \sqrt{\frac{ \sum_{i=0}^{n} pre_{i} - gt_{i} }{n}}
    :param pre: the predicted result 0-255
    :param gt: the ground truth 0-255
    :param ignore_zero: if 1, ignore 0 in gt, else 0, consider all pixel.
    :return: rmse computed
    """
    zero = tf.constant(0, dtype=tf.float32)
    if ignore_zero:
        where = tf.not_equal(gt, zero)
        indices = tf.where(where)
        pre = tf.gather_nd(pre, indices)
        gt = tf.gather_nd(gt, indices)

    nonzero_nums = tf.count_nonzero(gt)
    # if gt are all zero, set rmse = 0, else compute rmse
    rmse = tf.cond(
        tf.not_equal(
            nonzero_nums, tf.constant(
                0, dtype=tf.int64)), lambda: tf.sqrt(
            tf.reduce_mean(
                tf.square(
                    tf.subtract(
                        pre, gt)))), lambda: zero)
    return rmse
