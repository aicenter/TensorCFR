#!/usr/bin/env python3


def get_domain_by_name(name):
	from src.domains.domain01.Domain01 import get_domain01
	from src.domains.matching_pennies.MatchingPennies import get_domain_matching_pennies

	return {
		"domain01"        : get_domain01(),
		"matching_pennies": get_domain_matching_pennies(),
	}[name]