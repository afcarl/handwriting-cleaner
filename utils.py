"""Utilities for the project."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import itertools
import logging
import os
import re

import requests

import numpy as np
import six.moves.cPickle as pkl
import tensorflow as tf
import xml.etree.cElementTree as et


_MNIST_URL = 'https://s3.amazonaws.com/img-datasets/mnist.pkl.gz'
_HANDWRITING_URL = ('http://www.fki.inf.unibe.ch/databases/'
                    'iam-on-line-handwriting_tf-database')


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
    """Plots a handwriting_tf sample provided as a Numpy array.

    Args:
        sample: 2D Numpy array with shape (num_timesteps, 3), where the last
            dimension is (X, Y, penup), X and Y being the amount the pen
            changed in the X and Y directions from the previous point and
            penup being more than penup_threshold if it is the start of a new
            stroke.
        penup_threshold: float, if the penup feature is greater than this
            threshold that point is considered the start of a new stroke.
    """

    # The import is done here because of an issue with iOS virtual
    # environments.
    import matplotlib.pyplot as plt

    plt.gca().invert_yaxis()
    sample = np.copy(sample)
    sample[:, :2] = np.cumsum(sample[:, :2], axis=0)
    strokes = np.split(sample, np.where(sample[:, 2] > penup_threshold)[0][1:])
    for stroke in strokes:
        plt.plot(stroke[:, 0], stroke[:, 1])

    plt.gca().set_aspect('equal')
    plt.axis('off')


def get_data_path():
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
        name: str, the name of the file (e.g. mnist.pkl.gz) in the environment
            data path. If the name ends in .gz, it is interpretted
            as a gzipped file. If it ends in .tfrecords, it is interpretted as
            a tfrecords file.

    Returns:
        The loaded data. For exmaple, the MNIST data consists of two Numpy
            arrays.
    """

    file_path = os.path.join(get_data_path(), name)

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


def dump_file(name, obj):
    """Generic utility for saving datasets.

    Args:
        name: str, the name of the file in the environment data path. IF the
            file name ends in .gz, it is interpretted as a gzipped file.
        obj: the object to dump.
    """

    path = get_data_path()
    file_path = os.path.join(path, name)

    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'wb') as f:
            pkl.dump(obj, f)
    else:
        with open(file_path, 'w') as f:
            pkl.dump(obj, f)


def download_mnist(name='mnist.pkl.gz'):
    """Downloads and saves MNIST data in the data directory.

    Args:
        name: str, what name to save the MNIST data under.
    """

    path = get_data_path()
    file_path = os.path.join(path, name)

    if os.path.exists(file_path):
        raise ValueError('Already exists: %s' % file_path)

    response = requests.get(_MNIST_URL, stream=True)
    with open(file_path, "wb") as f:
        for data in response.iter_content():
            f.write(data)


def pad_to_len(sequence, target_length):
    """Pads the given sequence to the specified length.

    Args:
        sequence: N-D Tensor with shape (?, a1, a2...), the sequence to add
            padding. The first dimension should be the "length" dimension.
        target_length: int, the desired length to pad to.

    Returns:
        sequence_padded: N-D Tensor with shape (target_length, a1, a2...), the
            sequence padded with zeros at the end.
    """

    sequence_shape = sequence.get_shape().as_list()
    target_shape = [target_length] + list(sequence_shape[1:])

    if any(dim is None for dim in target_shape):
        raise ValueError('Invalid input sequence: All the dimensions except '
                         'the first one should be defined (got target shape '
                         '%s)' % target_shape)

    num_elements = reduce(lambda x, y: x * y, target_shape)
    as_vector = tf.reshape(sequence[:target_length], [-1])
    pad = tf.zeros([num_elements] - tf.shape(as_vector), dtype=sequence.dtype)
    vector_padded = tf.concat(0, [as_vector, pad])
    sequence_padded = tf.reshape(vector_padded, target_shape)

    return sequence_padded


def get_handwriting_arrays(data_file, num_timesteps, max_label_len):
    """Gets the input data as Numpy arrays.

    Args:
        data_file: str, the name of the input data file.
        num_timesteps: int, number of timesteps in the input.
        max_label_len: int, maximum number of characters in the label.

    Returns:
        strokes_array: 3D Numpy arrays with shape (num_samples, num_timesteps,
            3) with zero padding to get to num_timesteps.
        lines_array: 2D Numpy arrays with shape (num_samples, max_label_len)
            with zero padding to get to max_label_len.

    Raises:
        ValueError: if the data file has the wrong format.
    """

    if data_file.endswith('.tfrecords'):
        raise ValueError('Cannot use a TFRecords file for this. Got: %s'
                         % data_file)

    file_path = os.path.join(get_data_path(), data_file)

    if not os.path.exists(file_path):
        logging.info('File not found: %s. Creating it...', file_path)
        process_handwriting(data_file)

    strokes_list, lines_list = get_file(data_file)
    assert len(strokes_list) == len(lines_list)
    num_samples = len(strokes_list)

    strokes_array = np.zeros((num_samples, num_timesteps, 3), dtype=np.float32)
    lines_array = np.zeros((num_samples, max_label_len), dtype=np.int32)

    for i, (strokes, lines) in enumerate(zip(strokes_list, lines_list)):
        strokes = strokes[:num_timesteps]
        lines = lines[:max_label_len]
        strokes_array[i, :strokes.shape[0], :] = strokes
        lines_array[i, :lines.shape[0]] = lines

    return strokes_array, lines_array


def get_handwriting_tensors(data_file, batch_size, num_timesteps,
                            max_label_len):
    """Takes some input data and creates an input tensor with it.

    Args:
        data_file: str, the name of the input data file.
        batch_size: int, the size of a batch.
        num_timesteps: int, number of timesteps in the input.
        max_label_len: int, maximum number of characters in the label.

    Returns:
        strokes_tensor: 3D float Tensor (batch_size, num_timesteps, 3),
            the stroke deltas.
        stroke_len_tensor: 1D int Tensor (batch_size), the length of each
            stroke in the batch.
        label_tensor: 2D int Tensor (batch_size, label_len), the index-encoded
            labels (where each index represents a single character).
        label_len_tensor: 1D int Tensor (batch_size), the length of each label
            in the batch.

    Raises:
        ValueError: if the data file has the wrong format.
    """

    if not data_file.endswith('.tfrecords'):
        raise ValueError('The data file must be a TFRecords file. Got: %s'
                         % data_file)

    file_path = os.path.join(get_data_path(), data_file)

    if not os.path.exists(file_path):
        logging.info('File not found: %s. Creating it...', file_path)
        process_handwriting(data_file)

    filename_queue = tf.train.string_input_producer(
        [file_path], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    context_features = {
        'stroke_length': tf.FixedLenFeature([], dtype=tf.int64),
        'label_length': tf.FixedLenFeature([], dtype=tf.int64),
    }
    sequence_features = {
        'stroke': tf.VarLenFeature(dtype=tf.int64),
        'label': tf.VarLenFeature(dtype=tf.int64),
    }

    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features,
    )

    stroke_len_tensor = context_parsed['stroke_length']
    stroke_len_tensor = tf.minimum(stroke_len_tensor, num_timesteps)

    label_len_tensor = context_parsed['label_length']
    label_len_tensor = tf.minimum(label_len_tensor, max_label_len)

    # Makes sure the stroke_data tensor has the correct dimensions.
    stroke_tensor = tf.sparse_tensor_to_dense(sequence_parsed['stroke'])
    stroke_tensor = tf.transpose(stroke_tensor, (1, 0))
    stroke_tensor = tf.reshape(stroke_tensor, (-1, 3))
    stroke_tensor = pad_to_len(stroke_tensor, num_timesteps)

    # Makes sure the label_data tensor has the correct dimensions.
    label_tensor = tf.sparse_tensor_to_dense(sequence_parsed['label'])
    label_tensor = tf.reshape(label_tensor, (-1,))
    label_tensor = pad_to_len(label_tensor, max_label_len)

    # Adds batch part.
    (stroke_tensor, stroke_len_tensor, label_tensor,
     label_len_tensor) = tf.train.shuffle_batch(
        [stroke_tensor, stroke_len_tensor, label_tensor, label_len_tensor],
        batch_size=batch_size,
        capacity=2000,
        min_after_dequeue=1000)

    return stroke_tensor, stroke_len_tensor, label_tensor, label_len_tensor


def encode_handwriting_labels_as_indices(labels):
    """Encodes a list of labels as incides, building an appropriate dictionary.

    Args:
        labels: list of str, the labels to use.

    Returns:
        encoded_labels: list of Numpy arrays, the encoded labels, each with
            shape (characters_per_label, num_characters).
        char_to_idx: dictionary of char -> int pairs, the indices of each
            character.
    """

    all_chars = set((c for label in labels for c in label))
    char_to_idx = dict((c, i) for i, c in enumerate(all_chars))

    def _get_numpy_array(label):
        return np.asarray([char_to_idx[c] for c in label], dtype=np.int32)

    encoded_labels = [_get_numpy_array(label) for label in labels]
    return encoded_labels, char_to_idx


def process_handwriting(name,
                        dict_name='handwriting_dict.pkl.gz',
                        stroke_dir='data/lineStrokes',
                        label_dir='data/ascii',
                        num_samples=None):
    """Converts online handwriting_tf data to a Numpy arrays, and saves them.

    The handwriting_tf data can be downloaded by first registering an account here:
    http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting_tf-database

    In particular, you should download data/lineStrokes-all.tar.gz and
    data/ascii-all.tar.gz, then extract them to stroke_dir and label_dir
    respectively.

    The strokes and labels are saved as handwriting_tf proto objects. See
    protos/handwriting_tf.proto for the specific format. The TFRecords file that
    contains them can then be accessed using get_handwriting_input_tensors.

    Args:
        name: str, where to save the resulting file.
        dict_name: str, where to save the idx_to_char dictionary.
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
        """Parses the stroke data from the provided XML file.

        Args:
            stroke_path: str, the path to the XML file containing the stroke
                data.

        Returns:
            data: Numpy float array with shape (num_timesteps, 3), the data in
                the XML file as a Numpy array.
        """

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
        """Iterates through stroke and line data.

        Yields:
            stroke_path: str, the path to the file containing the data for the
                current stroke.
            line: str, the label for the current stroke.
        """

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

    # Gets the stroke paths and lines separately, then turns the lines into
    # Numpy arrays, saving the lookup dictionary to dict_name.
    stroke_paths, lines = zip(*iterable)
    lines_encoded, char_to_idx = encode_handwriting_labels_as_indices(lines)
    idx_to_char = dict((i, c) for c, i in char_to_idx.items())
    dump_file(dict_name, idx_to_char)

    file_path = os.path.join(get_data_path(), name)

    def _convert_to_sample(stroke_data, line_data):
        """Converts stroke and line data to a proto object."""

        stroke_len = tf.train.Int64List(value=[len(stroke_data)])
        label_len = tf.train.Int64List(value=[len(line_data)])
        x = tf.train.Int64List(value=stroke_data[:, 0].tolist())
        y = tf.train.Int64List(value=stroke_data[:, 1].tolist())
        penup = tf.train.Int64List(value=stroke_data[:, 2].tolist())
        label = tf.train.Int64List(value=line_data.tolist())

        sample = tf.train.SequenceExample(
            context=tf.train.Features(
                feature={
                    'stroke_length': tf.train.Feature(int64_list=stroke_len),
                    'label_length': tf.train.Feature(int64_list=label_len),
                }),
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    'stroke': tf.train.FeatureList(feature=[
                        tf.train.Feature(int64_list=x),
                        tf.train.Feature(int64_list=y),
                        tf.train.Feature(int64_list=penup),
                    ]),
                    'label': tf.train.FeatureList(feature=[
                        tf.train.Feature(int64_list=label)
                    ])
                }))

        return sample

    if file_path.endswith('.tfrecords'):
        with tf.python_io.TFRecordWriter(file_path) as writer:
            num_processed = 0

            for stroke_path, line_data in zip(stroke_paths, lines_encoded):
                stroke_data = _get_stroke_data(stroke_path)
                sample = _convert_to_sample(stroke_data, line_data)
                writer.write(sample.SerializeToString())

                num_processed += 1
                if num_processed % 100 == 0:
                    logging.info('Processed %d entries', num_processed)
    else:
        stroke_data = [_get_stroke_data(p) for p in stroke_paths]
        dump_file(name, (stroke_data, lines_encoded))
