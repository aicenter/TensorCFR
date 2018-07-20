import os

from src import utils
from src.algorithms.tensorcfr.TensorCFR import TensorCFR, get_cfr_strategies
from src.commons.constants import PROJECT_ROOT
from src.domains.available_domains import get_domain_by_name

if __name__ == '__main__':
	# domain = get_domain_by_name("matching_pennies")
	domain = get_domain_by_name("matching_pennies_via_gambit")
	tensorcfr = TensorCFR(domain)
	average_strategies = get_cfr_strategies(
			total_steps=10,
			tensorcfr_instance=tensorcfr,
			quiet=True,
			# profiling=True,
			delay=0
	)

	utils.gtlibrary.export_average_strategies_to_json(
		domain.information_set_mapping_to_gtlibrary,
		average_strategies,
		os.path.join(PROJECT_ROOT, 'matching_pennies_average_strategy.json'))
