from src.algorithms.tensorcfr_flattened_domains.TensorCFRFlattenedDomains import TensorCFRFlattenedDomains, get_cfr_strategies
from src.domains.available_domains import get_domain_by_name
from src.utils.other_utils import activate_script


if __name__ == '__main__' and activate_script():
	domain = get_domain_by_name("IIGS5_gambit_flattened")
	tensorcfr = TensorCFRFlattenedDomains(domain)
	get_cfr_strategies(
			total_steps=10,
			tensorcfr_instance=tensorcfr,
			quiet=True,
			# profiling=True,
			delay=0
	)   # TODO verify the results (final average strategies) via `gtlibrary`
