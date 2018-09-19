#!/usr/bin/env python3

import pandas as pd

if __name__ == '__main__':
	csv_basename = "IIGS3_1_3_false_true_lvl7"
	csv_filename = "{}.csv".format(csv_basename)
	features = pd.read_csv(
		csv_filename,
		names=["private_card1", "private_card2", "round1", "round2"],
		delimiter=";|,",
	)
	# TODO join two CSV files
	print("features:")
	print(features)

	print("###################################")

	sorted_features = features.sort_values(
		['round1', 'round2'],
		kind='mergesort'
	)
	print("sorted features:")
	print(sorted_features)

	# TODO reorder columns
	# TODO write to a CSV
