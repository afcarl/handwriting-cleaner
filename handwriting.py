"""Trains and evaluates handwriting model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import utils


def build_tf_records_file(path, data_file):
    """Builds a TF records file from handwriting data.

    Args:
        path: str, path to where the records file will be stored.
        data_file: str, the name of the Numpy data file.
    """

    stroke_data, label_data = utils.get_file(data_file)
    num_examples = len(stroke_data)

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    char_to_idx = dict()
    num_characters = 0

    writer = tf.python_io.TFRecordWriter(path)

    for index in range(num_examples):

        # Converts the label from a string to an int array.
        label_idx = list()
        for char in label_data[index]:
            if char not in char_to_idx:
                char_to_idx[char] = num_characters
                num_characters += 1
            label_idx.append(char_to_idx[char])

        converted_label = np.asarray(label_idx, np.int32)

        example = tf.train.Example(features=tf.train.Features(feature={
            'stroke': _int64_feature(stroke_data[index]),
            'label': _int64_feature(converted_label),
        }))
        writer.write(example.SerializeToString())
    writer.close()


def build_seq2seq_model(strokes_tensor, lens_tensor, labels_tensor):
    """Builds the seq2seq model.

    Args:
        strokes_tensor: 3D float Tensor (batch_size, num_timesteps, 3), where
            the last dimension is (X, Y, penup), the input data.
        lens_tensor: 1D int Tensor (batch_size), the length of each input
        labels_tensor: 2D int Tensor (batch_size, num_characters), the labels
            to go with the strokes, encoded by the dictionary.
    """

    batch_size = strokes_tensor.get_shape()[0].value
    max_timesteps = tf.reduce_max(lens_tensor)

    cells = [tf.nn.rnn_cell.GRUCell(128), tf.nn.rnn_cell.GRUCell(128)]
    seq_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    _, encoded_states = tf.nn.dynamic_rnn(seq_cell, strokes_tensor,
                                          sequence_length=lens_tensor,
                                          dtype=tf.float32,
                                          scope='encoder')

    with tf.variable_scope('decoder'):
        time = tf.constant(0, dtype=tf.int32)
        outputs_array = tf.TensorArray(tf.float32, size=max_timesteps)
        initial_stroke = tf.constant(0, tf.float32, shape=(batch_size, 3))

        def _decoder_step(time, prev_output, prev_rnn_states, outputs_array):
            new_output, new_rnn_states = seq_cell(prev_output, prev_rnn_states)
            new_output = utils.linear(new_output, 3)
            outputs_array = outputs_array.write(time, new_output)
            return time + 1, new_output, new_rnn_states, outputs_array

        _, _, _, rnn_outputs = tf.while_loop(
            cond=lambda time, *_: time < max_timesteps,
            body=_decoder_step,
            loop_vars=(time, initial_stroke, encoded_states, outputs_array))

        rnn_outputs = tf.transpose(rnn_outputs.pack(), [1, 0, 2])
        print('RNN outputs:', rnn_outputs.get_shape())


def main():
    #     build_tf_records_file(path='/tmp/handwriting.tfrecords',
    #                           data_file='handwriting_small.pkl.gz')
    batch_size = 8
    num_timesteps = 1000
    num_characters = 30

    strokes_tensor = tf.placeholder(tf.float32, (batch_size, num_timesteps, 3))
    lens_tensor = tf.placeholder(tf.int32, (batch_size,))
    labels_tensor = tf.placeholder(tf.int32, (batch_size, num_characters))
    build_seq2seq_model(strokes_tensor, lens_tensor, labels_tensor)


if __name__ == '__main__':
    main()
