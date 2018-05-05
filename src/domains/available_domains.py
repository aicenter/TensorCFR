#!/usr/bin/env python3
DOMAIN01 = "domain01"
MATCHING_PENNIES = "matching_pennies"
AVAILABLE_DOMAINS = [DOMAIN01, MATCHING_PENNIES]


def get_domain_by_name(name):
	if name == DOMAIN01:
		from src.domains.domain01.Domain01 import get_domain01
		return get_domain01()
	elif name == MATCHING_PENNIES:
		from src.domains.matching_pennies.MatchingPennies import get_domain_matching_pennies
		return get_domain_matching_pennies()
	else:
		raise ValueError("Invalid name '{}' for get_domain_by_name().".format(name))


if __name__ == '__main__':
	import tensorflow as tf

	for domain_name in AVAILABLE_DOMAINS:
		domain = get_domain_by_name(domain_name)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			domain.print_domain(session=sess)
