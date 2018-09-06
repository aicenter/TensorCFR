from src import utils
from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.algorithms.tensorcfr_flattened_domains.TensorCFRFlattenedDomains import TensorCFRFlattenedDomains, get_cfr_strategies
from src.domains.available_domains import get_domain_by_name

if __name__ == '__main__':
	domain = get_domain_by_name("II-GS2_gambit_flattened")
	tensorcfr = TensorCFRFixedTrunkStrategies(domain)
	average_strategies = tensorcfr.cfr_strategies_after_fixed_trunk(
		total_steps=1000,
		delay=0,
		register_strategies_on_step=[0, 1, 500, 999]
	)   # TODO verify the results (final average strategies) via `gtlibrary`
	# export average strategies to JSON
	utils.gtlibrary.export_average_strategies_to_json(
		domain,
		average_strategies,
		"GS2_average_strategies")
