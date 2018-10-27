#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd
import tensorflow as tf

from src.nn.features.goofspiel.IIGS6.game_constants import FEATURES_BASENAME, FEATURE_COLUMNS, TARGET_COLUMNS, \
	NAMES_OF_FEATURE_CSV
from src.utils.other_utils import get_files_in_directory_recursively, get_features_dataframe

ACTIVATE_FILE = False

FEATURES_PER_FILE = 128


def get_reaches_to_values_dataframe(filename):
	reaches_to_values_dataframe = pd.read_csv(
		filename,
		names=["nodal_index", "node_to_infoset", "nodal_reach", "nodal_expected_value"],
		delimiter=",",
		skiprows=1,
		index_col=0
	)
	return reaches_to_values_dataframe


def get_concatenated_dataframe(features_dataframe, reaches_to_values_dataframe):
	concatenated_dataframe = pd.concat(
		[features_dataframe, reaches_to_values_dataframe],
		axis=1,
	)[FEATURE_COLUMNS + TARGET_COLUMNS]
	return concatenated_dataframe


def get_sorted_dataframes(concatenated_dataframe):  # sort by lexicographically by public states and infosets
	sorted_dataframe = concatenated_dataframe.sort_values(
		FEATURE_COLUMNS,
		kind='mergesort'
	)
	print("sorted: ")
	print(sorted_dataframe.head())
	return sorted_dataframe


def verify_npz(filename, features, targets):
	dataset = np.load(filename)
	np.testing.assert_array_equal(features, dataset["features"])
	np.testing.assert_array_equal(targets, dataset["targets"])


def tfrecord_write(tfrecord_writer, data_input, data_target):
	n = data_target.shape[0]
	# loop through rows
	for i in range(n):
		# get one input and target
		dataset_sample_input = data_input[i, :]
		dataset_sample_target = data_target[i, :]
		dataset_sample_input = dataset_sample_input.tolist()
		dataset_sample_target = dataset_sample_target.tolist()
		# create Tensorflow's structures for the TFRecord
		dataset_sample = {
			'dataset_sample_input' : tf.train.Feature(float_list=tf.train.FloatList(value=dataset_sample_input)),
			'dataset_sample_target': tf.train.Feature(float_list=tf.train.FloatList(value=dataset_sample_target))
		}
		example = tf.train.Example(features=tf.train.Features(feature=dataset_sample))
		# write into the TFRecord
		tfrecord_writer.write(example.SerializeToString())


def prepare_dataset():
	"""
	:return: A boolean `True` if `dataset_dir` contains any files and `npz` is created.
	"""
	pd.set_option('display.max_columns', 500)
	pd.set_option('display.width', 1000)
	np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)

	script_directory = os.path.dirname(os.path.abspath(__file__))
	features_filename = "{}/{}.csv".format(script_directory, FEATURES_BASENAME)
	dataset_dir = "{}/csv_datasets/reach_value_datasets".format(script_directory)

	tfrecord_dataset_path = os.path.join(script_directory, "{}_{}".format('tfrecord_dataset', FEATURES_BASENAME))

	if not os.path.isdir(tfrecord_dataset_path):
		os.makedirs(tfrecord_dataset_path)

	features = get_features_dataframe(features_filename, NAMES_OF_FEATURE_CSV)
	filenames = get_files_in_directory_recursively(rootdir=dataset_dir)
	if not filenames:
		print("No files in {}".format(dataset_dir))
		return False
	else:
		# TFRecord's files counter
		tfrecord_file_cnt = 0
		# TFRecord's files writer
		tfrecord_writer = tf.python_io.TFRecordWriter(os.path.join(tfrecord_dataset_path, 'dataset_0.tfrecord'))
		# init temp arrays
		reach_arrays, target_arrays = list(), list()
		for i, reaches_to_values_filename in enumerate(filenames):
			print("#{}-th reaches_to_values_filename == {}".format(i, reaches_to_values_filename))
			reaches_to_values = get_reaches_to_values_dataframe(reaches_to_values_filename)
			concatenated = get_concatenated_dataframe(features, reaches_to_values)
			sorted_concatenated = get_sorted_dataframes(concatenated)

			np_dataset = sorted_concatenated.values
			reach_arrays.append(np_dataset[:, -2])
			target_arrays.append(np_dataset[:, -1])

			if (len(target_arrays) % FEATURES_PER_FILE) == 0:
				np_features = np.stack(reach_arrays)
				np_target = np.stack(target_arrays)

				# store to data into TFRecord file
				tfrecord_write(tfrecord_writer, np_features, np_target)

				# increment the TFRecord's files counter
				tfrecord_file_cnt += 1

				# re-init temp array with empty lists
				reach_arrays, target_arrays = list(), list()
				# close the TFRecord's writer and re-init for new TFRecord's file
				tfrecord_writer.close()
				tfrecord_writer = tf.python_io.TFRecordWriter(
					os.path.join(tfrecord_dataset_path, 'dataset_{}.tfrecord'.format(tfrecord_file_cnt)))
		# store the TFRecord file
		if len(reach_arrays) > 0 and len(target_arrays) > 0:
			np_features = np.stack(reach_arrays)
			np_target = np.stack(target_arrays)

			tfrecord_write(tfrecord_writer, np_features, np_target)
		# close TFRecord's file writer
		tfrecord_writer.close()

		return True


if __name__ == '__main__' and ACTIVATE_FILE:
	prepare_dataset()
