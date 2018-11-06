#!/usr/bin/env python3

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.domains.available_domains import get_domain_by_name

if __name__ == '__main__':
	domain = get_domain_by_name("II-GS3_gambit_flattened")
	tensorcfr = TensorCFRFixedTrunkStrategies(
		domain,
		trunk_depth=7
	)
	average_strategies = tensorcfr.cfr_strategies_after_fixed_trunk(
		total_steps=10,
		delay=2,
	)
