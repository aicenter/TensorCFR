#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd

from src.nn.features.goofspiel.IIGS3.game_constants import FEATURES_BASENAME, N_CARDS, SLICE_1HOT_FEATS, \
	NAMES_OF_FEATURE_CSV, FEATURE_COLUMNS
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

	if verbose:
		print("sorted_features:\n{}".format(sorted_features))
		print("np_features:\n{}".format(np_features))

	raise NotImplementedError

	# TODO
	one_hot_features = get_one_hot_flattened(
		features,   # TODO
		n_classes=N_CARDS,
		slice_1hot_feats=SLICE_1HOT_FEATS
	)


if __name__ == '__main__':
	get_1hot_round_card_features_np()
