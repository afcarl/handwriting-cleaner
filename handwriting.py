"""Trains and evaluates handwriting model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import utils


def encode_labels_as_indices(labels, char_to_idx=None):
    """Encodes a list of labels as incides, building an appropriate dictionary.

    Args:
        labels: list of str, the labels to use.
        char_to_idx: optional dict of char -> int pairs. If set, start with
            this look-up dict instead of initializing with a new one.

    Returns:
        encoded_labels: list of Numpy arrays, the encoded labels, each with
            shape (characters_per_label, num_characters).
        char_to_idx: updated dictionary of char -> int pairs, the indices of
            each character.
    """

    if char_to_idx is None:
        char_to_idx = dict()

    num_characters = len(char_to_idx)

    def _encode(character):
        if character not in char_to_idx:
            char_to_idx[character] = num_characters
            num_characters += 1
        return char_to_idx[character]

    encoded_labels = list()
    for label in labels:
        as_ints = [_encode(c) for c in label]
        encoded_labels = np.asarray(as_ints, dtype=np.int32)

    return encoded_labels, char_to_idx


def build_seq2seq_model(strokes_tensor, lens_tensor, labels_tensor):
    """Builds the seq2seq model.

    Args:
        strokes_tensor: 3D float Tensor (batch_size, num_timesteps, 3), where
            the last dimension is (X, Y, penup), the input data.
        lens_tensor: 1D int Tensor (batch_size), the length of each input
        labels_tensor: 2D int Tensor (batch_size, label_length), the labels
            to go with the strokes, encoded by the dictionary.
    """

    batch_size, num_timesteps, _ = strokes_tensor.get_shape()
    max_timesteps = tf.reduce_max(lens_tensor)

    cells = [tf.nn.rnn_cell.GRUCell(128), tf.nn.rnn_cell.GRUCell(128)]
    seq_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    _, encoded_states = tf.nn.dynamic_rnn(seq_cell, strokes_tensor,
                                          sequence_length=lens_tensor,
                                          dtype=tf.float32,
                                          scope='encoder')

    with tf.name_scope('variational_loss'):
        variational_loss = tf.constant(0., dtype=tf.float32)
        for encoded_state in encoded_states:
            mean, variance = tf.nn.moments(encoded_state, [1])
            variational_loss += tf.reduce_mean(tf.square(mean))
            variational_loss += tf.reduce_mean(tf.square(variance - 1))
        tf.scalar_summary('loss/variational', variational_loss)

    with tf.variable_scope('decoder'):
        time = tf.constant(0, dtype=tf.int32)
        outputs_array = tf.TensorArray(tf.float32, size=num_timesteps)
        initial_stroke = tf.constant(0, tf.float32, shape=(batch_size, 3))

        def _decoder_step(time, prev_output, prev_rnn_states, outputs_array):
            new_output, new_rnn_states = seq_cell(prev_output, prev_rnn_states)
            new_output = utils.linear(new_output, 3)
            new_output = tf.where(tf.greater(lens_tensor, time),
                                  new_output, tf.zeros_like(new_output))
            outputs_array = outputs_array.write(time, new_output)
            return time + 1, new_output, new_rnn_states, outputs_array

        _, _, _, rnn_outputs = tf.while_loop(
            cond=lambda time, *_: time < max_timesteps,
            body=_decoder_step,
            loop_vars=(time, initial_stroke, encoded_states, outputs_array))

        rnn_outputs = tf.transpose(rnn_outputs.pack(), [1, 0, 2])

    with tf.name_scope('reconstruction_loss'):
        reconstruction_loss = tf.square(rnn_outputs - strokes_tensor)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        tf.scalar_summary('loss/reconstruction', reconstruction_loss)

    total_loss = tf.add(variational_loss, reconstruction_loss, 'total_loss')
    tf.scalar_summary('loss/total', total_loss)


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
