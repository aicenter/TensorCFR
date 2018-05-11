from src.algorithms.tensorcfr.TensorCFR import TensorCFR, run_cfr
from src.domains.available_domains import get_domain_by_name

# TODO: Get rid of `ACTIVATE_FILE` hotfix in "#74 Storage for large files"
ACTIVATE_FILE = False

if __name__ == '__main__' and ACTIVATE_FILE:
	domain = get_domain_by_name("GP_cards4x3_222_via_gambit")
	tensorcfr = TensorCFR(domain)
	run_cfr(
			total_steps=10,
			tensorcfr_instance=tensorcfr,
			quiet=True,
			# profiling=True,
			delay=0
	)   # TODO verify the results (final average strategies) via `gtlibrary`
