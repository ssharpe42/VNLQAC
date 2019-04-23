import sys

import numpy as np
import tensorflow as tf


class MovingAvg(object):

    def __init__(self, p, burn_in=1):
        self.val = None
        self.p = p
        self.burn_in = burn_in

    def Update(self, v):
        if self.burn_in > 0:
            self.burn_in -= 1
            return v

        if self.val is None:
            self.val = v
            return v
        self.val = self.p * self.val + (1.0 - self.p) * v
        return self.val


def PrintParams(handle=sys.stdout.write):
    """Print the names of the parameters and their sizes.

    Args:
      handle: where to write the param sizes to
    """
    handle('NETWORK SIZE REPORT\n')
    param_count = 0
    fmt_str = '{0: <25}\t{1: >12}\t{2: >12,}\n'
    for p in tf.trainable_variables():
        shape = p.get_shape()
        shape_str = 'x'.join([str(x.value) for x in shape])
        handle(fmt_str.format(p.name, shape_str, np.prod(shape).value))
        param_count += np.prod(shape).value
    handle(''.join(['-'] * 60))
    handle('\n')
    handle(fmt_str.format('total', '', param_count))
    if handle == sys.stdout.write:
        sys.stdout.flush()


def GetRankInList(query, qlist):
    # returns the inverse rank of the item in the list
    indices = np.where(np.array(qlist) == query)[0]
    if len(indices) == 0:
        return 0
    else:
        return 1.0 / (1.0 + indices[0])


def prediction_threshold(prob, threshold=.5):
    return tf.cast(tf.math.greater(prob, threshold), tf.float32)


def true_positives(labels, predictions):
    return tf.reduce_sum(predictions * labels)


def false_positives(labels, predictions):
    return tf.reduce_sum(tf.clip_by_value(tf.subtract(predictions, labels), 0, 1))


def false_negatives(labels, predictions):
    return tf.reduce_sum(tf.clip_by_value(tf.subtract(labels, predictions), 0, 1))


#
# def recall(labels,predictions):
#
#     return tf.divide(tf.reduce_sum(predictions * labels),tf.reduce_sum(labels))
#
#
# def precision(labels,predictions):
#
#     return tf.divide(tf.reduce_sum(predictions * labels),tf.reduce_sum(predictions))

def recall(tp, fn):
    if tp == 0:
        return 0
    else:
        return 1.0 * tp / (tp + fn)


def precision(tp, fp):
    if tp == 0:
        return 0
    else:
        return 1.0 * tp / (tp + fp)


def f1(tp, fn, fp):
    r = recall(tp, fn)
    p = precision(tp, fp)
    if r * p == 0:
        return 0
    else:
        return 2.0 * r * p / (r + p)
