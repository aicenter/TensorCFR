from src import utils
from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.domains.available_domains import get_domain_by_name

if __name__ == '__main__':
	domain = get_domain_by_name("II-GS3_gambit_flattened")
	tensorcfr = TensorCFRFixedTrunkStrategies(domain)
	average_strategies = tensorcfr.cfr_strategies_after_fixed_trunk(
		total_steps=1000,
		delay=0,
		register_strategies_on_step=[1, 500, 999]
	)  # TODO verify the results (final average strategies) via `gtlibrary`

	utils.gtlibrary.export_average_strategies_to_json(
		domain,
		average_strategies,
		"GS3_average_strategies"
	)
