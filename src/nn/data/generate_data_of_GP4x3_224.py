#!/usr/bin/env python3
import os

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import SEED_FOR_TESTING, DEFAULT_DATASET_SIZE
from src.domains.available_domains import get_domain_by_name
from src.utils.other_utils import get_current_timestamp

# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = False


if __name__ == '__main__' and ACTIVATE_FILE:
	domain = get_domain_by_name("GP_cards4x3_224_gambit_flattened")
	tensorcfr = TensorCFRFixedTrunkStrategies(
		domain,
		trunk_depth=8
	)
	script_directory = os.path.dirname(os.path.abspath(__file__))
	print(get_current_timestamp())
	tensorcfr.generate_dataset_at_trunk_depth(
		dataset_size=DEFAULT_DATASET_SIZE,
		dataset_directory=script_directory + "/out",
		dataset_seed_to_start=SEED_FOR_TESTING
	)
	print(get_current_timestamp())
