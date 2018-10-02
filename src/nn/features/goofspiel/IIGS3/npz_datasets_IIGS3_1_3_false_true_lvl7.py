#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd

from src.utils.other_utils import get_files_in_directory_recursively, get_one_hot_flattened, get_features_dataframe

FEATURES_BASENAME = "IIGS3_1_3_false_true_lvl7"
N_CARDS = 3
FEATURE_COLUMNS = [
	"round1", "round2",
	"private_card1", "private_card2",
	"nodal_reach"
]
TARGET_COLUMNS = ["nodal_expected_value"]
SLICE_1HOT_FEATS = slice(4)


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
		[
			"round1", "round2",
			"private_card1", "private_card2",
		],
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
	np.set_printoptions(edgeitems=20, suppress=True)

	script_directory = os.path.dirname(os.path.abspath(__file__))
	features_filename = "{}/{}.csv".format(script_directory, FEATURES_BASENAME)
	dataset_dir = "{}/reach_value_datasets".format(script_directory)
	npz_filename = "{}/{}_numpy_dataset.npz".format(script_directory, FEATURES_BASENAME)

	features = get_features_dataframe(features_filename)
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

		np.savez_compressed(npz_filename, features=np_features, targets=np_targets)
		verify_npz(npz_filename, np_features, np_targets)
		return True


if __name__ == '__main__':
	prepare_dataset()
