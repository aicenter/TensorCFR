from src.algorithms.tensorcfr_flattened_domains.TensorCFRFlattenedDomains import TensorCFRFlattenedDomains, get_cfr_strategies
from src.domains.available_domains import get_domain_by_name


# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = False


if __name__ == '__main__' and ACTIVATE_FILE:
	import time
	print("Local current time (start): {}".format(
		time.asctime(
			time.localtime(
				time.time()
			)
		)
	))
	domain = get_domain_by_name("GP_cards4x3_224_gambit_flattened")
	tensorcfr = TensorCFRFlattenedDomains(domain)
	get_cfr_strategies(
			total_steps=10,
			tensorcfr_instance=tensorcfr,
			quiet=True,
			# profiling=True,
			delay=0
	)   # TODO verify the results (final average strategies) via `gtlibrary`
	print("Local current time (stop): {}".format(
		time.asctime(
			time.localtime(
				time.time()
			)
		)
	))