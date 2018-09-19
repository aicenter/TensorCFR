#!/usr/bin/env python3

import pandas as pd

if __name__ == '__main__':
	csv_basename = "IIGS6_1_6_false_true_lvl10"
	csv_filename = "{}.csv".format(csv_basename)
	features = pd.read_csv(
		csv_filename,
		names=["private_card1", "private_card2", "private_card3", "round1", "round2", "round3"],
		delimiter=";|,",
	)
	print(features)

	public_states_sizes = features.groupby(['round1', 'round2', 'round3']).size()
	csv_output_filename = "{}_public_states_sizes.csv".format(csv_basename)
	public_states_sizes.to_csv(
		csv_output_filename,
		header=True
	)
	print(public_states_sizes)
