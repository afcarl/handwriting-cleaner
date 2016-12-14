"""Utilities for the project."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import itertools
import os
import re

import requests

import matplotlib.pyplot as plt
import numpy as np
import six.moves.cPickle as pkl
import tensorflow as tf
import xml.etree.cElementTree as et


_MNIST_URL = 'https://s3.amazonaws.com/img-datasets/mnist.pkl.gz'
_HANDWRITING_URL = ('http://www.fki.inf.unibe.ch/databases/'
                    'iam-on-line-handwriting-database')


def linear(input_tensor, output_dims, activation=tf.tanh):
    """Applies a linear mapping on input_tensor to output_dims.

    If input_tensor has 2 dimensions (batch_size, input_dims), the linear
    mapping is applied across the last dimension to get a new Tensor with shape
    (batch_size, output_dims). If it has 3 dimensions (batch_size,
    num_timesteps, input_dims), the output tensor will have shape (batch_size,
    num_timesteps, output_dims).

    Args:
        input_tensor: a 2D or 3D float Tensor.
        output_dims: int, the number of output dimensions.
        activation: the activation function to use (e.g. tf.tanh, tf.sigmoid).

    Returns:
        A new Tensor with the same number of dimensions as the input Tensor,
        transformed as described above.

    Raises:
        ValueError: If the input tensor is poorly formed.
    """

    num_input_dims = len(input_tensor.get_shape())
    input_dims = input_tensor.get_shape()[-1].value

    if input_dims is None:
        raise ValueError('Invalid input tensor shape: %s' %
                         input_tensor.get_shape())

    w = tf.get_variable('W',
                        shape=(input_dims, output_dims),
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer(0., 1.))
    b = tf.get_variable('b',
                        shape=(output_dims,),
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0.0))

    if num_input_dims == 2:
        output_tensor = tf.nn.bias_add(tf.matmul(input_tensor, w), b)
    elif num_input_dims == 3:
        output_tensor = tf.map_fn(lambda x: tf.matmul(x, w), input_tensor)
        output_tensor = tf.nn.bias_add(output_tensor, b)
    else:
        raise ValueError('Invalid number of dimensions in the input tensor: '
                         '%d' % num_input_dims)

    output_tensor = activation(output_tensor)
    return output_tensor


def plot_handwriting_sample(sample, penup_threshold=0.5):
    """Plots a handwriting sample provided as a Numpy array.

    Args:
        sample: 2D Numpy array with shape (num_timesteps, 3), where the last
            dimension is (X, Y, penup), X and Y being the amount the pen
            changed in the X and Y directions from the previous point and
            penup being more than penup_threshold if it is the start of a new
            stroke.
        penup_threshold: float, if the penup feature is greater than this
            threshold that point is considered the start of a new stroke.
    """

    plt.gca().invert_yaxis()
    sample[:, :2] = np.cumsum(sample[:, :2], axis=0)
    strokes = np.split(sample, np.where(sample[:, 2] > penup_threshold)[0][1:])
    for stroke in strokes:
        plt.plot(stroke[:, 0], stroke[:, 1])

    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.show()


def _get_data_path():
    """Helper method for getting the data path in a user-friendly way."""

    if 'DATA_PATH' not in os.environ:
        raise ValueError('DATA_PATH environment variable must be set; it '
                         'should point the top-level directory where data is '
                         'stored.')

    if not os.path.exists(os.environ['DATA_PATH']):
        raise ValueError('The provided environment data path does not exist: '
                         '%s' % os.environ['DATA_PATH'])

    return os.environ['DATA_PATH']


def get_file(name):
    """Generic utility for loading datasets.

    Args:
        name: str, the name of the file (e.g. mnist.pkl.gz) in the directory
            specified by "path". If the name ends in .gz, it is interpretted
            as a gzipped file.

    Returns:
        The loaded data. For exmaple, the MNIST data consists of two Numpy
            arrays.
    """

    path = _get_data_path()
    file_path = os.path.join(path, name)

    if not os.path.exists(file_path):
        raise ValueError('No file found at %s. Use the scripts in utils.py '
                         'to download the required data.' % file_path)

    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rb') as f:
            data = pkl.load(f)
    else:
        with open(file_path, 'r') as f:
            data = pkl.load(f)

    return data


def download_mnist(name='mnist.pkl.gz'):
    """Downloads and saves MNIST data in the data directory.

    Args:
        name: str, what name to save the MNIST data under.
    """

    path = _get_data_path()
    file_path = os.path.join(path, name)

    if os.path.exists(file_path):
        raise ValueError('Already exists: %s' % file_path)

    response = requests.get(_MNIST_URL, stream=True)
    with open(file_path, "wb") as f:
        for data in response.iter_content():
            f.write(data)


def process_handwriting(name='handwriting.pkl.gz',
                        stroke_dir='data/lineStrokes',
                        label_dir='data/ascii',
                        num_samples=None):
    """Converts online handwriting data to a Numpy arrays, and saves them.

    The handwriting data can be downloaded by first registering an account here:
    http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database

    In particular, you should download data/lineStrokes-all.tar.gz and
    data/ascii-all.tar.gz, then extract them to stroke_dir and label_dir
    respectively.

    The stroke data is encoded as a Numpy array with dimensions (num_samples,
    sample_len, 3). The last dimension consists of (X, Y, penup), where X and Y
    are the deltas (how much the pen location changed from the previous step)
    and penup is 1 if is the start of a new stroke and 0 otherwise.

    Args:
        name: str, where to save the resulting file.
        stroke_dir: str, the path to the "lineStrokes" stroke data.
        label_dir: str, the path to the "ascii" labels data.
        num_samples: int, if specified, only process this many samples.

    Raises:
        ValueError: If the specified stroke_dir and label_dir don't exist.
    """

    if not os.path.exists(stroke_dir):
        raise ValueError('No stroke data exists at %s. It can be '
                         'downloaded from %s; download '
                         'data/lineStrokes-all.tar.gz and extract it to '
                         'the directory.' % (stroke_dir, _HANDWRITING_URL))

    if not os.path.exists(label_dir):
        raise ValueError('No stroke data exists at %s. It can be '
                         'downloaded from %s; download '
                         'data/ascii-all.tar.gz and extract it to '
                         'the directory.' % (label_dir, _HANDWRITING_URL))

    def _get_stroke_data(stroke_path):
        xml = et.parse(stroke_path)
        data = list()
        for stroke in xml.iter('Stroke'):
            for i, point in enumerate(stroke.iter('Point')):
                data.append([point.attrib['x'], point.attrib['y'], i == 0])
        data = np.asarray(data, dtype=np.int32)
        data[:, :2] = data[:, :2] - np.roll(data[:, :2], 1, axis=0)
        data[0, :2] = 0
        return data

    def _iter_strokes_and_lines():
        for root, _, files in os.walk(label_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                with open(file_path, 'r') as f:
                    file_text = f.read()

                matches = re.findall('CSR:(.+)$', file_text, re.DOTALL)
                if not matches:
                    continue

                assert len(matches) == 1, 'Malformed dataset'
                lines = matches[0].strip().replace('\r', '').split('\n')

                for i, line in enumerate(lines):
                    stroke_path = file_path.replace(
                        '.txt', '-%02d.xml' % (i + 1))
                    stroke_path = stroke_path.replace(label_dir, stroke_dir)
                    if not os.path.exists(stroke_path) or not line:
                        continue
                    yield stroke_path, line

    iterable = _iter_strokes_and_lines()
    if num_samples is not None:
        iterable = itertools.islice(iterable, num_samples)

    stroke_line_pairs = list()
    for stroke_path, line in iterable:
        stroke_data = _get_stroke_data(stroke_path)
        stroke_line_pairs.append((stroke_data, line))

    strokes, lines = zip(*stroke_line_pairs)

    path = _get_data_path()
    file_path = os.path.join(path, name)
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'w') as f:
            pkl.dump((strokes, lines), f)
    else:
        with open(file_path, 'w') as f:
            pkl.dump((strokes, lines), f)
