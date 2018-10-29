#!/usr/bin/env python3

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.domains import FlattenedDomain
from src.domains.available_domains import get_domain_by_name


class TensorCFR_BestResponse(TensorCFRFixedTrunkStrategies):
	def __init__(self, trunk_strategies, domain: FlattenedDomain, trunk_depth=0):
		super().__init__(domain, trunk_depth)
		self.trunk_strategies = trunk_strategies


if __name__ == '__main__':
	# domain_ = get_domain_by_name("flattened_hunger_games")
	# domain_ = get_domain_by_name("flattened_hunger_games_2")
	# domain_ = get_domain_by_name("flattened_domain01_via_gambit")
	# domain_ = get_domain_by_name("II-GS2_gambit_flattened")
	domain_ = get_domain_by_name("II-GS3_gambit_flattened")
	# domain_ = get_domain_by_name("IIGS5_gambit_flattened")
	# domain_ = get_domain_by_name("IIGS6_gambit_flattened")

	tensorcfr_instance = TensorCFR_BestResponse(
		domain_,
		trunk_depth=4
	)
	tensorcfr_instance.cfr_strategies_after_fixed_trunk(
		# total_steps=10,
		# storing_strategies=True,
		# profiling=True,
		# delay=0
		register_strategies_on_step=[1, 500, 999]
	)
