#!/usr/bin/env python3

DOMAIN01 = "domain01"
DOMAIN01_GAMBIT = "domain01_via_gambit"
FLATTENED_DOMAIN01_GAMBIT = "flattened_domain01_via_gambit"
MATCHING_PENNIES = "matching_pennies"
MATCHING_PENNIES_GAMBIT = "matching_pennies_via_gambit"
FLATTENED_MATCHING_PENNIES_GAMBIT = "flattened_matching_pennies_via_gambit"
HUNGER_GAMES = "hunger_games"
HUNGER_GAMES_2 = "hunger_games_2"
FLATTENED_HUNGER_GAMES = "flattened_hunger_games"
FLATTENED_HUNGER_GAMES_2 = "flattened_hunger_games_2"
GOOFSPIEL2_GAMBIT = "II-GS2_via_gambit"                   # for 13 cards
FLATTENED_GOOFSPIEL2_GAMBIT = "II-GS2_gambit_flattened"   # for 13 cards
FLATTENED_GOOFSPIEL2_2CARDS_GAMBIT = "II-GS2_2_cards_gambit_flattened"   # TODO
GOOFSPIEL3_GAMBIT = "II-GS3_via_gambit"
FLATTENED_GOOFSPIEL3_GAMBIT = "II-GS3_gambit_flattened"
FLATTENED_GOOFSPIEL3_SCALAR_UTIL_GAMBIT = "II-GS3_scalar_util_gambit_flattened"
GOOFSPIEL5_GAMBIT = "IIGS5_s1_bf_ft_via_gambit"
FLATTENED_GOOFSPIEL5_GAMBIT = "IIGS5_gambit_flattened"
GOOFSPIEL6_GAMBIT = "IIGS6_s1_bf_ft_via_gambit"
FLATTENED_GOOFSPIEL6_GAMBIT = "IIGS6_gambit_flattened"
PHANTOM_TTT_GAMBIT = "phantom_ttt_via_gambit"
PHANTOM_TTT_SINGLE_LEVEL_IS = "phantom_ttt_single_level_is"
GP_CARDS2X2_122 = "GP_cards2x2_122_via_gambit"
FLATTENED_GP_CARDS2X2_122 = "GP_cards2x2_122_gambit_flattened"
GP_CARDS4X3_222 = "GP_cards4x3_222_via_gambit"
FLATTENED_GP_CARDS4X3_222 = "GP_cards4x3_222_gambit_flattened"
GP_CARDS4X3_224 = "GP_cards4x3_224_via_gambit"
FLATTENED_GP_CARDS4X3_224 = "GP_cards4x3_224_gambit_flattened"
AVAILABLE_DOMAINS = [
	DOMAIN01,
	DOMAIN01_GAMBIT,
	FLATTENED_DOMAIN01_GAMBIT,
	MATCHING_PENNIES,
	MATCHING_PENNIES_GAMBIT,
	FLATTENED_MATCHING_PENNIES_GAMBIT,
	HUNGER_GAMES,
	HUNGER_GAMES_2,
	FLATTENED_HUNGER_GAMES,
	FLATTENED_HUNGER_GAMES_2,
	GOOFSPIEL2_GAMBIT,
	FLATTENED_GOOFSPIEL2_GAMBIT,
	GOOFSPIEL3_GAMBIT,
	FLATTENED_GOOFSPIEL3_GAMBIT,
	FLATTENED_GOOFSPIEL3_SCALAR_UTIL_GAMBIT
]
DOMAINS_WITH_LARGE_GAMBIT_FILES = [
	GOOFSPIEL5_GAMBIT,
	FLATTENED_GOOFSPIEL5_GAMBIT,
	GOOFSPIEL6_GAMBIT,
	FLATTENED_GOOFSPIEL6_GAMBIT,
	PHANTOM_TTT_GAMBIT,
	PHANTOM_TTT_SINGLE_LEVEL_IS,
	GP_CARDS2X2_122,
	FLATTENED_GP_CARDS2X2_122,
	GP_CARDS4X3_222,
	FLATTENED_GP_CARDS4X3_222,
	GP_CARDS4X3_224,
	FLATTENED_GP_CARDS4X3_224,
]
DOMAINS = AVAILABLE_DOMAINS + DOMAINS_WITH_LARGE_GAMBIT_FILES


def get_domain_by_name(name):
	if name == DOMAIN01:
		from src.domains.domain01.Domain01 import get_domain01
		return get_domain01()
	elif name == DOMAIN01_GAMBIT:
		from src.domains.domain01.domain_from_gambit_loader import get_domain01_from_gambit
		return get_domain01_from_gambit()
	elif name == FLATTENED_DOMAIN01_GAMBIT:
		from src.domains.flattened_domain01_gambit.domain_from_gambit_loader import get_flattened_domain01_from_gambit
		return get_flattened_domain01_from_gambit()
	elif name == MATCHING_PENNIES:
		from src.domains.matching_pennies.MatchingPennies import get_domain_matching_pennies
		return get_domain_matching_pennies()
	elif name == MATCHING_PENNIES_GAMBIT:
		from src.domains.matching_pennies.domain_from_gambit_loader import get_matching_pennies_from_gambit
		return get_matching_pennies_from_gambit()
	elif name == FLATTENED_MATCHING_PENNIES_GAMBIT:
		from src.domains.flattened_matching_pennies.flattened_matching_pennies_gambit \
			import get_flattened_matching_pennies_from_gambit
		return get_flattened_matching_pennies_from_gambit()
	elif name == HUNGER_GAMES:
		from src.domains.hunger_games.HungerGames import get_domain_hunger_games
		return get_domain_hunger_games()
	elif name == HUNGER_GAMES_2:
		from src.domains.hunger_games_2.HungerGames_2 import get_domain_hunger_games_2
		return get_domain_hunger_games_2()
	elif name == FLATTENED_HUNGER_GAMES:
		from src.domains.flattened_hunger_games.FlattenedHungerGames import get_flattened_domain_hunger_games
		return get_flattened_domain_hunger_games()
	elif name == FLATTENED_HUNGER_GAMES_2:
		from src.domains.flattened_hunger_games_2.FlattenedHungerGames_2 import get_flattened_domain_hunger_games_2
		return get_flattened_domain_hunger_games_2()
	elif name == GOOFSPIEL2_GAMBIT:
		from src.domains.goofspiel2.domain_from_gambit_loader import get_domain_goofspiel2
		return get_domain_goofspiel2()
	elif name == FLATTENED_GOOFSPIEL2_GAMBIT:
		from src.domains.flattened_goofspiel2.domain_from_gambit_loader import get_flattened_goofspiel2
		return get_flattened_goofspiel2()
	elif name == FLATTENED_GOOFSPIEL3_GAMBIT:
		from src.domains.flattened_goofspiel3.domain_from_gambit_loader import get_flattened_goofspiel3
		return get_flattened_goofspiel3()
	elif name == FLATTENED_GOOFSPIEL3_SCALAR_UTIL_GAMBIT:
		from src.domains.flattened_goofspiel3_scalar_util.domain_from_gambit_loader import get_flattened_goofspiel3_scalar_util
		return get_flattened_goofspiel3_scalar_util()
	elif name == GOOFSPIEL3_GAMBIT:
		from src.domains.goofspiel3.domain_from_gambit_loader import get_domain_goofspiel3
		return get_domain_goofspiel3()
	elif name == GOOFSPIEL5_GAMBIT:
		from src.domains.goofspiel5.domain_from_gambit_loader import get_domain_goofspiel5
		return get_domain_goofspiel5()
	elif name == FLATTENED_GOOFSPIEL5_GAMBIT:
		from src.domains.flattened_goofspiel5.domain_from_gambit_loader import get_flattened_goofspiel5
		return get_flattened_goofspiel5()
	elif name == GOOFSPIEL6_GAMBIT:
		from src.domains.goofspiel6.domain_from_gambit_loader import get_domain_goofspiel6
		return get_domain_goofspiel6()
	elif name == FLATTENED_GOOFSPIEL6_GAMBIT:
		from src.domains.flattened_goofspiel6.domain_from_gambit_loader import get_flattened_goofspiel6
		return get_flattened_goofspiel6()
	elif name == PHANTOM_TTT_GAMBIT:
		from src.domains.phantom_ttt.domain_from_gambit_loader import get_domain_phantom_ttt
		return get_domain_phantom_ttt()
	elif name == PHANTOM_TTT_SINGLE_LEVEL_IS:
		from src.domains.phantom_ttt.domain_from_gambit_loader import get_domain_phantom_ttt_single_level_IS
		return get_domain_phantom_ttt_single_level_IS()
	elif name == GP_CARDS2X2_122:
		from src.domains.GP_cards2x2_122.domain_from_gambit_loader import get_domain_GP_cards2x2_122
		return get_domain_GP_cards2x2_122()
	elif name == FLATTENED_GP_CARDS2X2_122:
		from src.domains.flattened_GP_cards2x2_122.domain_from_gambit_loader import get_flattened_GP_cards2x2_122
		return get_flattened_GP_cards2x2_122()
	elif name == GP_CARDS4X3_222:
		from src.domains.GP_cards4x3_222.domain_from_gambit_loader import get_domain_GP_cards4x3_222
		return get_domain_GP_cards4x3_222()
	elif name == FLATTENED_GP_CARDS4X3_222:
		from src.domains.flattened_GP_cards4x3_222.domain_from_gambit_loader import get_flattened_GP_cards4x3_222
		return get_flattened_GP_cards4x3_222()
	elif name == GP_CARDS4X3_224:
		from src.domains.GP_cards4x3_224.domain_from_gambit_loader import get_domain_GP_cards4x3_224
		return get_domain_GP_cards4x3_224()
	elif name == FLATTENED_GP_CARDS4X3_224:
		from src.domains.flattened_GP_cards4x3_224.domain_from_gambit_loader import get_flattened_GP_cards4x3_224
		return get_flattened_GP_cards4x3_224()
	else:
		raise ValueError("Invalid name '{}' for get_domain_by_name().".format(name))


if __name__ == '__main__':
	import tensorflow as tf

	for domain_name in AVAILABLE_DOMAINS:
		domain = get_domain_by_name(domain_name)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			domain.print_domain(session=sess)
