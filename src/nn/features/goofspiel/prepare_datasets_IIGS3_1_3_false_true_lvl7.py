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


def load_from_npz(dataset_filename, np_features, np_targets):
	dataset = np.load(dataset_filename)
	print("diff between reloaded features:")
	print(np_features - dataset["features"])
	print("diff between reloaded targets:")
	print(np_targets - dataset["targets"])


def save_to_npz(pandas_dataframe, dataset_filename):
	np_dataset = pandas_dataframe.values
	print(np_dataset)
	print("features:")
	np_features = np_dataset[:, :-1]
	print(np_features)
	print("targets:")
	np_targets = np_dataset[:, -1]
	print(np_targets)
	np.savez_compressed(dataset_filename, features=np_features, targets=np_targets)
	load_from_npz(dataset_filename, np_features, np_targets)


if __name__ == '__main__':
	pd.set_option('display.max_columns', 500)
	pd.set_option('display.width', 1000)
	script_directory = os.path.dirname(os.path.abspath(__file__))
	features_basename = "IIGS3_1_3_false_true_lvl7"
	features = get_features_dataframe()
	filenames = get_files_in_directory_recursively(rootdir="{}/reach_value_datasets".format(script_directory))

	n_files = len(filenames)
	n_nodes = features.shape[0]
	feature_dim = len(FEATURE_COLUMNS)
	target_dim = len(TARGET_COLUMNS)
	print("{} files x {} nodes x {} feature_dim".format(n_files, n_nodes, feature_dim))
	np_features = np.zeros((n_files, n_nodes, feature_dim))  # shape [#seed_of_the_batch, #nodes, #features]
	print("{} files x {} nodes x {} target_dim".format(n_files, n_nodes, target_dim))
	np_targets = np.zeros((n_files, n_nodes, target_dim))  # shape [#seed_of_the_batch, #nodes, #targets]

	for reaches_to_values_filename in filenames:
		print("reaches_to_values_filename == {}".format(reaches_to_values_filename))
		reaches_to_values = get_reaches_to_values_dataframe()
		concatenated = get_concatenated_dataframe(features, reaches_to_values)
		sorted_concatenated = get_sorted_dataframes(concatenated)
		# save_to_npz(
		# 	sorted_concatenated,
		# 	dataset_filename="{}/{}_numpy_dataset.npz".format(script_directory, features_basename)
		# )
