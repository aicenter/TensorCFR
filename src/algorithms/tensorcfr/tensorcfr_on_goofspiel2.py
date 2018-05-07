from src.algorithms.tensorcfr.TensorCFR import TensorCFR, run_cfr
from src.domains.available_domains import get_domain_by_name

if __name__ == '__main__':
	domain = get_domain_by_name("II-GS2_via_gambit")
	tensorcfr = TensorCFR(domain)
	run_cfr(
			total_steps=10,
			tensorcfr_instance=tensorcfr,
			quiet=True,
			# profiling=True,
			delay=0
	)
