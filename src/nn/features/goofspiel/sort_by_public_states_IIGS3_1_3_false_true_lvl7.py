#!/usr/bin/env python3

import os

import pandas as pd

if __name__ == '__main__':
	script_directory = os.path.dirname(os.path.abspath(__file__))
	pd.set_option('display.max_columns', 500)
	pd.set_option('display.width', 1000)

	features_basename = "IIGS3_1_3_false_true_lvl7"
	features_filename = "{}/{}.csv".format(script_directory, features_basename)
	features = pd.read_csv(
		features_filename,
		names=["private_card1", "private_card2", "round1", "round2"],
		delimiter=";|,",
	)
	print("features:")
	print(features)

	print("###################################")

	seed = 0
	reaches_to_values_basename = "nodal_dataset_seed_{}".format(seed)
	reaches_to_values_filename = "{}/{}.csv".format(script_directory, reaches_to_values_basename)
	print("reaches_to_values_filename == {}".format(reaches_to_values_filename))
	reaches_to_values = pd.read_csv(
		reaches_to_values_filename,
		names=["nodal_index", "node_to_infoset", "nodal_reach", "nodal_expected_value"],
		delimiter=",",
		skiprows=1
	)
	print("reaches_to_values:")
	print(reaches_to_values)

	print("###################################")

	concatenated = pd.concat(
		[features, reaches_to_values],
		axis=1,
		# sort=True
	)
	print("concatenated:")
	print(concatenated)

	print("###################################")

	sorted_features = features.sort_values(
		['round1', 'round2'],
		kind='mergesort'
	)
	print("sorted features:")
	print(sorted_features)

	# TODO reorder columns
	# TODO write to a CSV
