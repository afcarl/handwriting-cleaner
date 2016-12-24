"""Trains and evaluates handwriting model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf
import utils


def build_seq2seq_autoencoder(strokes_tensor, lens_tensor, global_step):
    """Builds the seq2seq model.

    Args:
        strokes_tensor: 3D float Tensor (batch_size, num_timesteps, 3), where
            the last dimension is (X, Y, penup), the input data.
        lens_tensor: 1D int Tensor (batch_size), the length of each input
        global_step: scalar int Tensor representing the global step.
    """

    batch_size, _, _ = strokes_tensor.get_shape()
    max_timesteps = tf.reduce_max(lens_tensor)

    cells = [tf.nn.rnn_cell.GRUCell(128), tf.nn.rnn_cell.GRUCell(128)]
    seq_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    _, encoded_states = tf.nn.dynamic_rnn(seq_cell, strokes_tensor,
                                          sequence_length=lens_tensor,
                                          dtype=tf.float32,
                                          scope='encoder')

    with tf.name_scope('variational_loss'):
        variational_losses = list()
        for encoded_state in encoded_states:
            mean, variance = tf.nn.moments(encoded_state, [1])
            variational_losses.append(tf.reduce_mean(tf.square(mean)))
            variational_losses.append(tf.reduce_mean(tf.square(variance - 1)))
        variational_loss = tf.reduce_sum(variational_losses)
        tf.scalar_summary('loss/variational', variational_loss)

    with tf.variable_scope('decoder'):
        time = tf.constant(0, dtype=tf.int32)
        outputs_array = tf.TensorArray(tf.float32, size=max_timesteps)
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
        strokes_tensor = strokes_tensor[:, :max_timesteps]

    with tf.name_scope('reconstruction_loss'):
        reconstruction_loss = tf.square(rnn_outputs - strokes_tensor)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        tf.scalar_summary('loss/reconstruction', reconstruction_loss)

    total_loss = tf.add(variational_loss, reconstruction_loss, 'total_loss')
    tf.scalar_summary('loss/total', total_loss)

    optimizer = tf.train.AdamOptimizer()
    grads, train_vars = zip(*optimizer.compute_gradients(total_loss))

    clipped_grads, _ = tf.clip_by_global_norm(grads, 0.1)
    train_op = optimizer.apply_gradients(
        zip(clipped_grads, train_vars), global_step)

    return train_op


def start_worker(cluster, server):
    """Starts one worker node with the given cluster / server configuration."""

    FLAGS = tf.app.flags.FLAGS

    with tf.name_scope('inputs'):
        strokes, stroke_lens, _, _ = utils.get_handwriting_tensors(
            FLAGS.data_file,
            FLAGS.batch_size,
            FLAGS.max_timesteps,
            FLAGS.max_label_len)

        # Casts the input tensors to the correct types.
        strokes = tf.cast(strokes, tf.float32)
        stroke_lens = tf.cast(stroke_lens, tf.int32)

        # Applies L2 normalization along the time dimension.
        strokes = tf.nn.l2_normalize(strokes, dim=1)

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

    train_op = build_seq2seq_autoencoder(strokes,
                                         stroke_lens,
                                         global_step)

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir=FLAGS.logdir,
                             global_step=global_step,
                             save_model_secs=600,
                             save_summaries_secs=120)

    with sv.managed_session(server.target) as sess:
        while not sv.should_stop():
            _, step = sess.run([train_op, global_step])
            logging.info('step: %d' % step)

    sv.stop()


def train():
    """Loads data, builds the model and trains it."""

    FLAGS = tf.app.flags.FLAGS

    with tf.Graph().as_default():

        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.worker_hosts.split(',')
        cluster = tf.train.ClusterSpec({"ps": ps_hosts,
                                        "worker": worker_hosts})

        if FLAGS.run_local:
            server = tf.train.Server.create_local_server()
        else:
            server = tf.train.Server(cluster,
                                     job_name=FLAGS.job_name,
                                     task_index=FLAGS.task_index)

        if FLAGS.job_name == 'ps':
            server.join()
        elif FLAGS.job_name == 'worker':
            if FLAGS.run_local:
                start_worker(cluster, server)
            else:
                with tf.device(tf.train.replica_device_setter(
                        worker_device="/job:worker/task:%d" % FLAGS.task_index,
                        cluster=cluster)):
                    start_worker(cluster, server)
        else:
            raise ValueError('Invalid job name: "%s" (should be either "ps" '
                             'or "worker")' % FLAGS.job_name)


def main():
    tf.app.flags.DEFINE_integer('batch_size', 8, 'Size of each batch.')
    tf.app.flags.DEFINE_integer('max_timesteps', 700, 'Maximum number of '
                                'timesteps in each handwriting sample.')
    tf.app.flags.DEFINE_integer('max_label_len', 50, 'Maximum length of a '
                                'label.')
    tf.app.flags.DEFINE_string('data_file', 'handwriting.tfrecords', 'Name of '
                               'the TFRecords file with the handwriting data.')

    tf.app.flags.DEFINE_string('logdir', '/tmp/handwriting', 'Directory to '
                               'write logs.')
    tf.app.flags.DEFINE_bool('run_local', True, 'If set, run the server '
                             'locally instead of in a distributed setting.')

    # Flags for defining the ClusterSpec.
    tf.app.flags.DEFINE_string('ps_hosts', '',
                               'Comma-separated list of hostname:port pairs')
    tf.app.flags.DEFINE_string('worker_hosts', '',
                               'Comma-separated list of hostname:port pairs')

    # Flags for defining the Server.
    tf.app.flags.DEFINE_string('job_name', '', 'One of ["ps", "worker"], the '
                               'type of job to start.')
    tf.app.flags.DEFINE_integer('task_index', 0, 'Index of this task.')

    # Makes the logger show everything.
    logging.getLogger('').setLevel(logging.INFO)

    train()


if __name__ == '__main__':
    main()
