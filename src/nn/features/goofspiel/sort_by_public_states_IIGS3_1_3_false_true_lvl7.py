#!/usr/bin/env python3

import os

import pandas as pd

if __name__ == '__main__':
	script_directory = os.path.dirname(os.path.abspath(__file__))
	features_basename = "{}/IIGS3_1_3_false_true_lvl7".format(script_directory)
	features_filename = "{}.csv".format(features_basename)
	features = pd.read_csv(
		features_filename,
		names=["private_card1", "private_card2", "round1", "round2"],
		delimiter=";|,",
	)
	print("features:")
	print(features)

	# TODO join two CSV files

	print("###################################")

	sorted_features = features.sort_values(
		['round1', 'round2'],
		kind='mergesort'
	)
	print("sorted features:")
	print(sorted_features)

# TODO reorder columns
# TODO write to a CSV
