#!/usr/bin/env python3

import pandas as pd

if __name__ == '__main__':
	features = pd.read_csv(
		"IIGS3_features.csv",
		names=["private_card1", "private_card2", "round1", "round2"],
		# delimiter=";|,",
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
