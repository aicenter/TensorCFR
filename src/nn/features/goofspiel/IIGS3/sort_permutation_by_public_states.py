#!/usr/bin/env python3

import pandas as pd

from src.nn.features.goofspiel.IIGS3.game_constants import FEATURE_COLUMNS, TARGET_COLUMNS

ACTIVATE_FILE = True


def get_reaches_to_values_dataframe(filename):
	reaches_to_values_dataframe = pd.read_csv(
		filename,
		names=["nodal_index", "node_to_infoset", "nodal_reach", "nodal_expected_value"],
		delimiter=",",
		skiprows=1,
		index_col=0
	)
	return reaches_to_values_dataframe


def get_concatenated_dataframe(features_dataframe, reaches_to_values_dataframe):
	concatenated_dataframe = pd.concat(
		[features_dataframe, reaches_to_values_dataframe],
		axis=1,
	)[FEATURE_COLUMNS + TARGET_COLUMNS]
	return concatenated_dataframe


def get_sorted_dataframes(concatenated_dataframe):  # sort by lexicographically by public states and infosets
	sorted_dataframe = concatenated_dataframe.sort_values(
		FEATURE_COLUMNS,
		kind='mergesort'
	)
	print("sorted: ")
	print(sorted_dataframe.head())
	return sorted_dataframe


def get_permutation_by_public_states():
	raise NotImplementedError


if __name__ == '__main__' and ACTIVATE_FILE:
	get_permutation_by_public_states()
