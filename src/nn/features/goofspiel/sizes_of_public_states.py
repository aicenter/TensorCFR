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
	print(features)
	public_states_sizes = features.groupby(['round1', 'round2']).size()
	print(public_states_sizes)
