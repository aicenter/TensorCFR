from src.algorithms.tensorcfr.TensorCFR import TensorCFR, run_cfr
from src.domains.available_domains import get_domain_by_name, MATCHING_PENNIES, MATCHING_PENNIES_GAMBIT

if __name__ == '__main__':
	domain = get_domain_by_name(MATCHING_PENNIES)
	# domain = get_domain_by_name(MATCHING_PENNIES_GAMBIT)
	tensorcfr = TensorCFR(domain)
	run_cfr(
			# total_steps=10,
			tensorcfr_instance=tensorcfr,
			quiet=True
	)
