from src.algorithms.tensorcfr.TensorCFR import TensorCFR, run_cfr
from src.domains.available_domains import get_domain_by_name

ACTIVATE_FILE = False


if __name__ == '__main__' and ACTIVATE_FILE:
	domain = get_domain_by_name("IIGS5_s1_bf_ft_via_gambit")
	tensorcfr = TensorCFR(domain)
	run_cfr(
			total_steps=10,
			tensorcfr_instance=tensorcfr,
			quiet=True,
			# profiling=True,
			delay=0
	)
