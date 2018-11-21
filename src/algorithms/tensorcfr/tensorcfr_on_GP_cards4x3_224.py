from src.algorithms.tensorcfr.TensorCFR import TensorCFR, get_cfr_strategies
from src.domains.available_domains import get_domain_by_name
from src.utils.other_utils import activate_script


if __name__ == '__main__' and activate_script():
	domain = get_domain_by_name("GP_cards4x3_224_via_gambit")
	tensorcfr = TensorCFR(domain)
	get_cfr_strategies(
			total_steps=10,
			tensorcfr_instance=tensorcfr,
			quiet=True,
			# profiling=True,
			delay=0
	)   # TODO verify the results (final average strategies) via `gtlibrary`
