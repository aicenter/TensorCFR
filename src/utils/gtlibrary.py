import json


def export_average_strategies_to_json(information_set_mapping_to_gtlibrary, average_strategies, output_file):
	return_json = [None] * len(information_set_mapping_to_gtlibrary)

	for _ in information_set_mapping_to_gtlibrary:
		index_gtlibrary = information_set_mapping_to_gtlibrary[_]["gtlibrary_index"]
		level, index_in_level = information_set_mapping_to_gtlibrary[_]["tensorcfr_strategy_coordination"]


		# convert a Numpy 1D array to a list of floats
		strategy = average_strategies[level][index_in_level].tolist()
		return_json[index_gtlibrary] = strategy

	with open(output_file, 'w') as f:
		json.dump(return_json, f)

	return True