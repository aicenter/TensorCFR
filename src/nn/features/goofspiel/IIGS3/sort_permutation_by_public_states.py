#!/usr/bin/env python3

import numpy as np
import pandas as pd

from src.nn.features.goofspiel.IIGS3.game_constants import FEATURE_COLUMNS, FEATURES_BASENAME, \
	NAMES_OF_FEATURE_CSV
from src.utils.other_utils import get_features_dataframe


def get_permutation_by_public_states():
	"""
	:return: The permutation that sorts game nodes of IIGS3 at level 7 by their public states and infosets.
	"""
	pd.set_option('display.max_columns', 500)
	pd.set_option('display.width', 1000)
	np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)

	features_filename = "{}.csv".format(FEATURES_BASENAME)
	features = get_features_dataframe(features_filename, NAMES_OF_FEATURE_CSV, quiet=True)
	features = features[FEATURE_COLUMNS[:-1]]
	print(features)


if __name__ == '__main__':
	get_permutation_by_public_states()
