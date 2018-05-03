#!/usr/bin/env python3

from src.algorithms.tensorcfr.TensorCFR import TensorCFR
from src.domains.domain01.Domain01 import domain01
from src.domains.matching_pennies.MatchingPennies import matching_pennies

if __name__ == '__main__':
	for tensorcfr in [TensorCFR(domain01), TensorCFR(matching_pennies)]:
		# tensorcfr.run_cfr(total_steps=10, delay=0)
		# tensorcfr.run_cfr(total_steps=10, delay=0, quiet=True)
		# tensorcfr.run_cfr()
		# from src.commons.constants import DEFAULT_TOTAL_STEPS_ON_SMALL_DOMAINS
		# tensorcfr.run_cfr(total_steps=DEFAULT_TOTAL_STEPS_ON_SMALL_DOMAINS, delay=5)
		tensorcfr.run_cfr(quiet=True)
		# tensorcfr.run_cfr(quiet=True, total_steps=10000)
