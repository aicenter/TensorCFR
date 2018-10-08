#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd

from src.nn.features.goofspiel.IIGS3.game_constants import FEATURES_BASENAME, N_CARDS, FEATURE_COLUMNS, TARGET_COLUMNS, \
	SLICE_1HOT_FEATS, NAMES_OF_FEATURE_CSV
from src.utils.other_utils import get_files_in_directory_recursively, get_one_hot_flattened, get_features_dataframe


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


def prepare_dataset():
	"""
	:return: A boolean `True` if `dataset_dir` contains any files and `npz` is created.
	"""
	pd.set_option('display.max_columns', 500)
	pd.set_option('display.width', 1000)
	np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)

	script_directory = os.path.dirname(os.path.abspath(__file__))
	features_filename = "{}/{}.csv".format(script_directory, FEATURES_BASENAME)
	dataset_dir = "{}/reach_value_datasets".format(script_directory)
	npz_basename = "{}/{}".format(script_directory, FEATURES_BASENAME)

	features = get_features_dataframe(features_filename, NAMES_OF_FEATURE_CSV)
	filenames = get_files_in_directory_recursively(rootdir=dataset_dir)
	if not filenames:
		print("No files in {}".format(dataset_dir))
		return False
	else:
		feature_arrays, target_arrays = [], []
		for i, reaches_to_values_filename in enumerate(filenames):
			print("#{}-th reaches_to_values_filename == {}".format(i, reaches_to_values_filename))
			reaches_to_values = get_reaches_to_values_dataframe(reaches_to_values_filename)
			concatenated = get_concatenated_dataframe(features, reaches_to_values)
			sorted_concatenated = get_sorted_dataframes(concatenated)

			np_dataset = sorted_concatenated.values
			feature_arrays.append(np_dataset[:, :-1])
			target_arrays.append(np_dataset[:, -1])

		raw_features = np.stack(feature_arrays)  # shape [#seed_of_the_batch, #nodes, #features]
		one_hot_features = get_one_hot_flattened(
			raw_features,
			n_classes=N_CARDS,
			slice_1hot_feats=SLICE_1HOT_FEATS
		)
		reaches = np.expand_dims(raw_features[..., -1], axis=-1)
		np_features = np.concatenate((one_hot_features, reaches), axis=-1)
		print("np_features:\n{}".format(np_features))
		print("np_features.shape == {}".format(np_features.shape))

		np_targets = np.stack(target_arrays)  # shape [#seed_of_the_batch, #nodes]
		print("np_targets:\n{}".format(np_targets))
		print("np_targets.shape == {}".format(np_targets.shape))

		# split dataset to train/dev/test
		trainset_ratio = 0.8
		devset_ratio = 0.1
		dataset_size = len(filenames)
		split_train = int(trainset_ratio * dataset_size)
		split_dev = int((trainset_ratio + devset_ratio) * dataset_size)

		train_features, train_targets = np_features[:split_train], np_targets[:split_train]
		dev_features, dev_targets = np_features[split_train:split_dev], np_targets[split_train:split_dev]
		test_features, test_targets = np_features[split_dev:], np_targets[split_dev:]

		# store trainset
		train_file = "{}_train.npz".format(npz_basename)
		np.savez_compressed(train_file, features=train_features, targets=train_targets)
		verify_npz(train_file, train_features, train_targets)

		# store devset
		dev_file = "{}_dev.npz".format(npz_basename)
		np.savez_compressed(dev_file, features=dev_features, targets=dev_targets)
		verify_npz(dev_file, dev_features, dev_targets)

		# store testset
		test_file = "{}_test.npz".format(npz_basename)
		np.savez_compressed(test_file, features=test_features, targets=test_targets)
		verify_npz(test_file, test_features, test_targets)

		return True


if __name__ == '__main__':
	prepare_dataset()
