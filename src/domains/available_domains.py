#!/usr/bin/env python3


def get_domain_by_name(name):
	if name == "domain01":
		from src.domains.domain01.Domain01 import get_domain01
		return get_domain01()
	elif name == "matching_pennies":
		from src.domains.matching_pennies.MatchingPennies import get_domain_matching_pennies
		return get_domain_matching_pennies()
	else:
		raise ValueError("Invalid name '{}' for get_domain_by_name().".format(name))


# TODO fill-in __main__
