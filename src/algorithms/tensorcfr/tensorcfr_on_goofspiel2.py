import json

from src.algorithms.tensorcfr.TensorCFR import TensorCFR, run_cfr
from src.domains.available_domains import get_domain_by_name


if __name__ == '__main__':
	domain = get_domain_by_name("II-GS2_via_gambit")
	tensorcfr = TensorCFR(domain)
	average_strategies = run_cfr(
			total_steps=100,
			tensorcfr_instance=tensorcfr,
			quiet=True,
			# profiling=True,
			delay=0
	)   # TODO verify the results (final average strategies) via `gtlibrary`

	# for i in range(len(average_strategies)):
	# 	print("level " + str(i))
	# 	print(average_strategies[i])
	#
	# print("DOMAIN MAPPING")
	# print(domain.information_set_mapping_to_gtlibrary)
	#
	# json_to_gtlibrary = [None] * len(domain.information_set_mapping_to_gtlibrary)
	#
	# for _ in domain.information_set_mapping_to_gtlibrary:
	# 	index_gtlibrary = domain.information_set_mapping_to_gtlibrary[_]["gtlibrary_index"]
	# 	level = domain.information_set_mapping_to_gtlibrary[_]["tensorcfr_strategy_coordination"][0]
	# 	index_in_level = domain.information_set_mapping_to_gtlibrary[_]["tensorcfr_strategy_coordination"][1]
	#
	# 	# convert Numpy 1D array to list of floats
	# 	strategy = average_strategies[level][index_in_level].tolist()
	# 	json_to_gtlibrary[index_gtlibrary] = strategy
	#
	# import pprint
	# pprint.pprint(json_to_gtlibrary)
	#
	# with open("/home/ruda/IdeaProjects/gtlibrary/tensorcfr_strategy_4_BR.json", 'w') as f:
	# 	json.dump(json_to_gtlibrary, f)


