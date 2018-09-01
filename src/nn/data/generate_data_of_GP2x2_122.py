#!/usr/bin/env python3
import os

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import SEED_FOR_TESTING, DEFAULT_DATASET_SIZE
from src.domains.available_domains import get_domain_by_name
from src.utils.other_utils import get_current_timestamp

if __name__ == '__main__':
	domain = get_domain_by_name("GP_cards2x2_122_gambit_flattened")
	tensorcfr = TensorCFRFixedTrunkStrategies(
		domain,
		trunk_depth=5
	)
	script_directory = os.path.dirname(os.path.abspath(__file__))
	print(get_current_timestamp())
	tensorcfr.generate_dataset_multiple_sessions(
		dataset_size=DEFAULT_DATASET_SIZE,
		dataset_directory=script_directory + "/out",
		dataset_seed_to_start=SEED_FOR_TESTING
	)
	print(get_current_timestamp())
