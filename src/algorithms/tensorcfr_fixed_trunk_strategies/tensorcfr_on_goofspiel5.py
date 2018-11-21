from src import utils
from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.domains.available_domains import get_domain_by_name
from src.utils.other_utils import activate_script


if __name__ == '__main__' and activate_script():
	domain = get_domain_by_name("IIGS5_gambit_flattened")
	tensorcfr = TensorCFRFixedTrunkStrategies(
		domain,
		trunk_depth=0
	)
	average_strategies = tensorcfr.cfr_strategies_after_fixed_trunk(
		total_steps=1000,
		delay=0,
		register_strategies_on_step=[0, 250, 500, 750, 999]
	)  # TODO verify the results (final average strategies) via `gtlibrary`

	utils.gtlibrary.export_average_strategies_to_json(
		domain,
		average_strategies,
		"GS5_average_strategies"
	)
