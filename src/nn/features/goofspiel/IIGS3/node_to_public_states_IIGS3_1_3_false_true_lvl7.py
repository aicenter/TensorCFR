#!/usr/bin/env python3
import os

import numpy as np
from pandas import DataFrame

from src.utils.other_utils import get_features_dataframe


def get_node_to_public_state(verbose=False):
	script_directory = os.path.dirname(os.path.abspath(__file__))
	csv_basename = "IIGS3_1_3_false_true_lvl7"
	csv_filename = os.path.join(script_directory, "{}.csv".format(csv_basename))
	features = get_features_dataframe(csv_filename)

	grouped_features = features.groupby(['round1', 'round2'])
	df_public_states_sizes = DataFrame(
		{"public_state_size": (grouped_features.size())}
	).reset_index()
	np_public_states_sizes = df_public_states_sizes["public_state_size"].values
	node_to_public_state = np.array([
		i
		for i, size in enumerate(np_public_states_sizes)
		for _ in range(size)
	])

	if verbose:
		print("grouped_features:\n{}.".format(grouped_features.head()))
		print("df_public_states_sizes:\n{}\n".format(df_public_states_sizes))
		print("np_public_states_sizes:\n{}\n".format(np_public_states_sizes))
		print("node_to_public_state:\n{}\n".format(node_to_public_state))

	return node_to_public_state


if __name__ == '__main__':
	node_to_public_state_mapping = get_node_to_public_state(verbose=True)
	print("node_to_public_state_mapping:\n{}\n".format(node_to_public_state_mapping))
