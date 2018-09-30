#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd

from src.utils.other_utils import get_files_in_directory_recursively

FEATURE_COLUMNS = [
	"round1", "round2",
	"private_card1", "private_card2",
	"node_to_infoset",
	"nodal_reach"
]
TARGET_COLUMNS = ["nodal_expected_value"]


def get_features_dataframe():
	features_filename = "{}/{}.csv".format(script_directory, features_basename)
	features_dataframe = pd.read_csv(
		features_filename,
		names=["private_card1", "private_card2", "round1", "round2"],
		delimiter=";|,",
	)
	print("features:")
	print(features_dataframe)
	return features_dataframe


def get_reaches_to_values_dataframe():
	reaches_to_values_dataframe = pd.read_csv(
		reaches_to_values_filename,
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


def get_sorted_dataframes(concatenated_dataframe):
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


if __name__ == '__main__':
	pd.set_option('display.max_columns', 500)
	pd.set_option('display.width', 1000)
	script_directory = os.path.dirname(os.path.abspath(__file__))
	features_basename = "IIGS3_1_3_false_true_lvl7"
	features = get_features_dataframe()
	filenames = get_files_in_directory_recursively(rootdir="{}/reach_value_datasets".format(script_directory))

	feature_arrays, target_arrays = [], []
	for i, reaches_to_values_filename in enumerate(filenames):
		print("#{}-th reaches_to_values_filename == {}".format(i, reaches_to_values_filename))
		reaches_to_values = get_reaches_to_values_dataframe()
		concatenated = get_concatenated_dataframe(features, reaches_to_values)
		sorted_concatenated = get_sorted_dataframes(concatenated)

		np_dataset = sorted_concatenated.values
		feature_arrays.append(np_dataset[:, :-1])
		target_arrays.append(np_dataset[:, -1])

	np_features = np.stack(feature_arrays)  # shape [#seed_of_the_batch, #nodes, #features]
	print(np_features)
	print("np_features.shape == {}".format(np_features.shape))
	np_targets = np.stack(target_arrays)  # shape [#seed_of_the_batch, #nodes]
	print(np_targets)
	print("np_targets.shape == {}".format(np_targets.shape))

	dataset_filename = "{}/{}_numpy_dataset.npz".format(script_directory, features_basename)
	np.savez_compressed(dataset_filename, features=np_features, targets=np_targets)
	verify_npz(dataset_filename, np_features, np_targets)
