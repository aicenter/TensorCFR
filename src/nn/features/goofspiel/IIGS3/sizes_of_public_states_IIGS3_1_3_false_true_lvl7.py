#!/usr/bin/env python3
import os

from src.utils.other_utils import get_features_dataframe

if __name__ == '__main__':
	script_directory = os.path.dirname(os.path.abspath(__file__))
	csv_basename = "IIGS3_1_3_false_true_lvl7"
	csv_filename = os.path.join(script_directory, "{}.csv".format(csv_basename))
	features = get_features_dataframe(csv_filename)  # TODO

	public_states_sizes = features.groupby(['round1', 'round2']).size()
	csv_output_filename = "{}_public_states_sizes.csv".format(csv_basename)
	public_states_sizes.to_csv(
		csv_output_filename,
		header=True
	)
	print(public_states_sizes)
