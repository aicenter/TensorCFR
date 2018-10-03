#!/usr/bin/env python3
import os

import numpy as np
import pandas as pd

from src.utils.other_utils import get_features_dataframe

if __name__ == '__main__':
	pd.set_option('display.max_columns', 500)
	pd.set_option('display.width', 1000)
	np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)

	script_directory = os.path.dirname(os.path.abspath(__file__))
	csv_basename = "IIGS6_1_6_false_true_lvl10"
	csv_filename = os.path.join(script_directory, "{}.csv".format(csv_basename))
	features = get_features_dataframe(csv_filename)   # TODO change feature names

	grouped_features = features.groupby(['round1', 'round2', 'round3'])
	print("grouped_features:\n{}.".format(grouped_features.head()))
	public_states_sizes = grouped_features.size()
	csv_output_filename = "{}_public_states_sizes.csv".format(csv_basename)
	public_states_sizes.to_csv(
		csv_output_filename,
		header=True
	)
	print(public_states_sizes)
