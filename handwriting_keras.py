"""Trains and evaluates Keras handwriting model.

I switched over from using the TensorFlow model after thinking it would be a
good idea to try using Keras.js (https://github.com/transcranial/keras-js).
This way the files could be hosted statically and I could put it on my website
without having to set up a server.

The model is a seq2seq autoencoder, which learns to take a sample and generate
itself. Ideally, it would learn to represent words in the cleanest way, so that
when a messy sample is fed to the model, the autoencoder reinterprets it as a
clean version.

Should probably also look at D3.js for something.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os

import keras

import matplotlib.pyplot as plt
import numpy as np
import utils


def build_seq2seq_autoencoder(num_timesteps):
    """Builds the seq2seq model.

    Args:
        num_timesteps: int, number of input timesteps.
    """

    input_var = keras.layers.Input(shape=(num_timesteps, 3))

    # Builds encoder part.
    conv_enc_1 = keras.layers.Convolution1D(64, 2)(input_var)
    conv_enc_2 = keras.layers.Convolution1D(64, 2)(conv_enc_1)
    latent_vec = keras.layers.wrappers.Bidirectional(
        keras.layers.GRU(64))(conv_enc_2)

    # Batch normalization on the latent vector is probably a good idea.
    latent_vec = keras.layers.BatchNormalization(axis=1)(latent_vec)

    # Builds decoder part.
    repeated_vec = keras.layers.RepeatVector(num_timesteps)(latent_vec)
    rnn_dec_1 = keras.layers.wrappers.Bidirectional(
        keras.layers.GRU(64, return_sequences=True))(repeated_vec)
    output_var = keras.layers.TimeDistributed(keras.layers.Dense(3))(rnn_dec_1)
    output_var = keras.layers.Activation('tanh')(output_var)

    model = keras.models.Model(input=[input_var], output=[output_var])
    model.compile(loss='mae', optimizer='adam')

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_timesteps', type=int, default=700)
    parser.add_argument('--max_label_len', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='output')
    parser.add_argument('--data_file', type=str, default='handwriting.pkl.gz')
    args = parser.parse_args()

    # Makes the logger show everything.
    logging.getLogger('').setLevel(logging.INFO)

    strokes, _ = utils.get_handwriting_arrays(args.data_file,
                                              args.num_timesteps,
                                              args.max_label_len)

    eps = 1e-12
    strokes = strokes / (np.linalg.norm(strokes, axis=1, keepdims=True) + eps)
    eval_set = strokes[:5]

    # Builds the model itself.
    model = build_seq2seq_autoencoder(args.num_timesteps)

    def _save_sample_callback(epoch, _):
        """Plots the model's output on the current epoch and saves the plot."""

        model_preds = model.predict(eval_set)

        plt.figure(1)
        for i in range(5):
            plt.subplot(5, 2, 1 + 2 * i)
            utils.plot_handwriting_sample(eval_set[i], penup_threshold=0.0)
            plt.title('Real Sample %d' % (i + 1))

            plt.subplot(5, 2, 2 + 2 * i)
            utils.plot_handwriting_sample(model_preds[i], penup_threshold=0.0)
            plt.title('Real Sample %d' % (i + 1))

        plt.savefig(os.path.join(args.logdir, 'epoch_%d_results.png' % epoch))
        plt.close()

    callbacks = [
        keras.callbacks.ModelCheckpoint(args.logdir,
                                        save_best_only=True,
                                        save_weights_only=True),
        keras.callbacks.LambdaCallback(on_epoch_begin=_save_sample_callback),
    ]

    with open(os.path.join(args.logdir, 'model.json'), 'w') as f:
        f.write(model.to_json())

    model.fit(x=[strokes], y=[strokes], nb_epoch=100,
              validation_split=0.2, callbacks=callbacks)


if __name__ == '__main__':
    main()
