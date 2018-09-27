#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd


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


def get_concatenated_dataframe():
	concatenated_dataframe = pd.concat(
		[features, reaches_to_values],
		axis=1,
	)[[
		"round1", "round2",
		"private_card1", "private_card2",
		"node_to_infoset",
		"nodal_reach",
		"nodal_expected_value"
	]]
	print("concatenated:")
	print(concatenated_dataframe)
	return concatenated_dataframe


def get_sorted_dataframes():
	sorted_dataframe = concatenated.sort_values(
		[
			"round1", "round2",
			"private_card1", "private_card2",
		],
		kind='mergesort'
	)
	print("sorted: ")
	print(sorted_dataframe)
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

	print("###################################")
	seed = 0
	reaches_to_values_basename = "nodal_dataset_seed_{}".format(seed)
	reaches_to_values_filename = "{}/{}.csv".format(script_directory, reaches_to_values_basename)
	print("reaches_to_values_filename == {}".format(reaches_to_values_filename))
	reaches_to_values = pd.read_csv(
		reaches_to_values_filename,
		names=["nodal_index", "node_to_infoset", "nodal_reach", "nodal_expected_value"],
		delimiter=",",
		skiprows=1,
		index_col=0
	)
	print("reaches_to_values:")
	print(reaches_to_values)
	print("###################################")
	concatenated = get_concatenated_dataframe()
	print("###################################")
	sorted_concatenated = get_sorted_dataframes()
	print("###################################")
	save_to_npz(
		sorted_concatenated,
		dataset_filename="{}/{}_numpy_dataset.npz".format(script_directory, features_basename)
	)
