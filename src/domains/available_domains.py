#!/usr/bin/env python3
DOMAIN01 = "domain01"
MATCHING_PENNIES = "matching_pennies"
DOMAIN01_GAMBIT = "domain01_via_gambit"
MATCHING_PENNIES_GAMBIT = "matching_pennies_via_gambit"
GOOFSPIEL2_GAMBIT = "II-GS2_via_gambit"
GOOFSPIEL3_GAMBIT = "II-GS3_via_gambit"
GOOFSPIEL5_GAMBIT = "IIGS5_s1_bf_ft_via_gambit"
PHANTOM_TTT_GAMBIT = "phantom_ttt_via_gambit"
AVAILABLE_DOMAINS = [
	DOMAIN01,
	MATCHING_PENNIES,
	DOMAIN01_GAMBIT,
	MATCHING_PENNIES_GAMBIT,
	GOOFSPIEL2_GAMBIT,
	GOOFSPIEL3_GAMBIT,
]
DOMAINS_WITH_LARGE_GAMBIT_FILES = [
	GOOFSPIEL5_GAMBIT,
	PHANTOM_TTT_GAMBIT,
]
DOMAINS = AVAILABLE_DOMAINS + DOMAINS_WITH_LARGE_GAMBIT_FILES


def get_domain_by_name(name):
	if name == DOMAIN01:
		from src.domains.domain01.Domain01 import get_domain01
		return get_domain01()
	elif name == MATCHING_PENNIES:
		from src.domains.matching_pennies.MatchingPennies import get_domain_matching_pennies
		return get_domain_matching_pennies()
	elif name == DOMAIN01_GAMBIT:
		from src.domains.domain01.domain_from_gambit_loader import get_domain01_from_gambit
		return get_domain01_from_gambit()
	elif name == MATCHING_PENNIES_GAMBIT:
		from src.domains.matching_pennies.domain_from_gambit_loader import get_matching_pennies_from_gambit
		return get_matching_pennies_from_gambit()
	elif name == GOOFSPIEL2_GAMBIT:
		from src.domains.goofspiel2.domain_from_gambit_loader import get_domain_goofspiel2
		return get_domain_goofspiel2()
	elif name == GOOFSPIEL3_GAMBIT:
		from src.domains.goofspiel_3.domain_from_gambit_loader import get_domain_goofspiel_3
		return get_domain_goofspiel_3()
	elif name == GOOFSPIEL5_GAMBIT:
		from src.domains.goofspiel_5.domain_from_gambit_loader import get_domain_goofspiel_5
		return get_domain_goofspiel_5()
	elif name == PHANTOM_TTT_GAMBIT:
		from src.domains.phantom_ttt.domain_from_gambit_loader import get_domain_phantom_ttt
		return get_domain_phantom_ttt()
	else:
		raise ValueError("Invalid name '{}' for get_domain_by_name().".format(name))


if __name__ == '__main__':
	import tensorflow as tf

	for domain_name in AVAILABLE_DOMAINS:
		domain = get_domain_by_name(domain_name)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			domain.print_domain(session=sess)
