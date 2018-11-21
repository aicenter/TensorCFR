from src import utils
from src.algorithms.tensorcfr_flattened_domains.TensorCFRFlattenedDomains import TensorCFRFlattenedDomains, get_cfr_strategies
from src.domains.available_domains import get_domain_by_name
from src.utils.other_utils import activate_script


if __name__ == '__main__' and activate_script():
	domain = get_domain_by_name("IIGS6_gambit_flattened")
	tensorcfr = TensorCFRFlattenedDomains(domain)
	average_strategies = get_cfr_strategies(
			total_steps=1000,
			tensorcfr_instance=tensorcfr,
			quiet=True,
			# profiling=True,
			delay=0,
			register_strategies_on_step=[0, 250, 500, 750, 999]
	)   # TODO verify the results (final average strategies) via `gtlibrary`

	utils.gtlibrary.export_average_strategies_to_json(
		domain,
		average_strategies,
		"GS6_average_strategies")