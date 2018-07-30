from src.algorithms.tensorcfr.TensorCFR import TensorCFR, get_cfr_strategies
from src.domains.available_domains import get_domain_by_name

# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = False


if __name__ == '__main__' and ACTIVATE_FILE:
	domain = get_domain_by_name("IIGS5_s1_bf_ft_via_gambit")
	tensorcfr = TensorCFR(domain)
	get_cfr_strategies(
			total_steps=10,
			tensorcfr_instance=tensorcfr,
			quiet=True,
			# profiling=True,
			delay=0
	)   # TODO verify the results (final average strategies) via `gtlibrary`
