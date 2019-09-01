from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tarfile
from six.moves import cPickle as pickle
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

dataset_items = {
        'cifar10': {
            'filename': 'cifar-10-python.tar.gz',
            'download_url': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'local_folder': 'cifar-10-batches-py',
            'image_shape': (32, 32, 3),
            'num_class': 10,
            }
        }

def get_label_names(dataset_type, data_dir):
    if not dataset_items.get(dataset_type, None):
        print("Invalid dataset_type, valid types are", ",".join(dataset_items.keys()))
        return
    local_folder = dataset_items[dataset_type]['local_folder']
    class_dict =  read_pickle_from_file(
            os.path.join(
                os.path.join(
                    data_dir,
                    local_folder
                    ),
                'batches.meta'
                )
            )
    return class_dict[b'label_names']

def download_and_extract(data_dir, filename, download_url):
    """download dataset if not already downloaded"""
    tf.contrib.learn.datasets.base.maybe_download(filename, data_dir, download_url)
    tarfile.open(os.path.join(data_dir, filename), 'r:gz').extractall(data_dir)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _get_file_names():
    """Returns the file names expected to exist in the input_dir"""
    file_names = {}
    file_names['train'] = ['data_batch_%d' % i for i in xrange(1, 6)]
    file_names['eval'] = ['test_batch']
    return file_names

def read_pickle_from_file(filename):
    with tf.io.gfile.GFile(filename, 'rb') as f:
        if sys.version_info >= (3, 0):
            data_dict = pickle.load(f, encoding='bytes')
        else:
            data_dict = pickle.load(f)
    return data_dict

def convert_to_tfrecord(input_files, output_file, output_prefix):
    """Converts a file to TFRecords."""
    print('Generating %s...' % output_file, end='')
    count = 0
    with tf.io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            data_dict = read_pickle_from_file(input_file)
            data = data_dict[b'data']
            labels = data_dict[b'labels']
            num_entries_in_batch = len(labels)
            for i in range(num_entries_in_batch):
                example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'image': _bytes_feature(data[i].tobytes()),
                        'label': _int64_feature(labels[i])
                    }))
                record_writer.write(example.SerializeToString())
            count += num_entries_in_batch
    print('Done [%d records]' % (len(input_files)*num_entries_in_batch))

def create(dataset_type, data_dir):
    """
    Read dataset data from pickled numpy arrays and writes TFRecords.
    Generates tf.train.Example protos and writes them to TFRecord files from the
    python version of the dataset
    """
    if not dataset_items.get(dataset_type, None):
        print("Invalid dataset_type, valid types are", ",".join(dataset_items.keys()))
        return
    file_names = _get_file_names()
    create = False
    for f_name in file_names.keys():
        if not os.path.exists(data_dir + "/" + f_name + '.tfrecords'):
            create = True
    if not create:
        print('TFRecords already exists in ' + data_dir)
        return
    dataset_filename = dataset_items[dataset_type]['filename']
    download_url = dataset_items[dataset_type]['download_url']
    local_folder = dataset_items[dataset_type]['local_folder']
    print('Download from {} and extract.'.format(download_url))
    download_and_extract(data_dir, dataset_filename, download_url)
    input_dir = os.path.join(data_dir, local_folder)
    for mode, files in sorted(file_names.items(), key=lambda item: item[0]):
        input_files = [os.path.join(input_dir, f) for f in files]
        output_file = os.path.join(data_dir, mode + '.tfrecords')
        try:
            os.remove(output_file)
        except OSError:
            pass
        # Convert to tf.train.Example and write the to TFRecords.
        convert_to_tfrecord(input_files, output_file, mode)
    print('Done!')

def load(dataset_type, filenames, batch_size, preprocess_fn, training=False, one_hot_label=True):
    if not dataset_items.get(dataset_type, None):
        print("Invalid dataset_type, valid types are", ",".join(dataset_items.keys()))
        return

    img_height, img_width, img_depth = dataset_items[dataset_type]['image_shape']
    num_classes = dataset_items[dataset_type]['num_class']

    def _parse_record(serialized_example):
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),
            }
        )

        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([img_depth * img_height * img_width])
        image = tf.reshape(image, [img_depth, img_height, img_width])
        image = tf.cast(tf.transpose(image, [1, 2, 0]), tf.float32)

        label = tf.cast(features['label'], tf.int32)
        if one_hot_label:
            label = tf.one_hot(label, num_classes)
        return image, label

    dataset = tf.data.TFRecordDataset(filenames=filenames)

    if training:
      buffer_size = batch_size * 2 + 1
      dataset = dataset.shuffle(buffer_size=buffer_size)

    # Transformation
    dataset = dataset.map(_parse_record, num_parallel_calls=4)
    img_shape = (img_height, img_width, img_depth)
    dataset = dataset.map(lambda image, label: (preprocess_fn(image, img_shape, training), label))

    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2 * batch_size)
    return dataset
