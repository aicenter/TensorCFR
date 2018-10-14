#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd

from src.nn.features.goofspiel.IIGS3.game_constants import FEATURES_BASENAME, N_CARDS, NAMES_OF_FEATURE_CSV, \
	FEATURE_COLUMNS, N_ROUNDS, N_ROUND_RESULTS
from src.utils.other_utils import get_one_hot_flattened, get_features_dataframe


def get_1hot_round_card_features_np(verbose=True):
	"""
	:return: A numpy array of 1-hot encoded features of round results and cards.
	"""
	pd.set_option('display.max_columns', 500)
	pd.set_option('display.width', 1000)
	np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)

	script_directory = os.path.dirname(os.path.abspath(__file__))
	features_filename = "{}/{}.csv".format(script_directory, FEATURES_BASENAME)

	features = get_features_dataframe(features_filename, NAMES_OF_FEATURE_CSV)
	one_hot_columns = FEATURE_COLUMNS[:-1]
	sorted_features = features.sort_values(
		one_hot_columns,
		kind='mergesort'
	)[one_hot_columns]
	np_features = sorted_features.values
	np_rounds = np_features[:, :N_ROUNDS]
	np_cards = np_features[:, N_ROUNDS:]

	one_hot_rounds = get_one_hot_flattened(
		np_rounds,
		n_classes=N_ROUND_RESULTS,
		slice_1hot_feats=slice(N_ROUNDS)
	)
	one_hot_cards = get_one_hot_flattened(
		np_cards,
		n_classes=N_CARDS,
		slice_1hot_feats=slice(2 * N_ROUNDS)    # N_ROUNDS per each of the 2 players
	)

	if verbose:
		print("sorted_features:\n{}".format(sorted_features))
		print("np_features:\n{}".format(np_features))
		print("np_rounds:\n{}".format(np_rounds))
		print("np_cards:\n{}".format(np_cards))
		print("one_hot_rounds:\n{}".format(one_hot_rounds))
		print("one_hot_cards:\n{}".format(one_hot_cards))


if __name__ == '__main__':
	get_1hot_round_card_features_np()
