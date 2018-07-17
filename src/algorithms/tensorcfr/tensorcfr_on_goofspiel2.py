from src.algorithms.tensorcfr.TensorCFR import TensorCFR, run_cfr
from src.domains.available_domains import get_domain_by_name


if __name__ == '__main__':
	domain = get_domain_by_name("II-GS2_via_gambit")
	tensorcfr = TensorCFR(domain)
	average_strategies = run_cfr(
			total_steps=10,
			tensorcfr_instance=tensorcfr,
			quiet=True,
			# profiling=True,
			delay=0
	)   # TODO verify the results (final average strategies) via `gtlibrary`

	for i in range(len(average_strategies)):
		print("level " + str(i))
		print(average_strategies[i])
