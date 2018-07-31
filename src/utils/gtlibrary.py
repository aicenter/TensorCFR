import datetime
import os
import json


def export_average_strategies_to_json(information_set_mapping_to_gtlibrary, average_strategies, output_filename):
	output_dir = "out"
	return_json = [None] * len(information_set_mapping_to_gtlibrary)

	for mapping in information_set_mapping_to_gtlibrary:
		if "gtlibrary_index" not in information_set_mapping_to_gtlibrary[mapping]:
			return False

		if "tensorcfr_strategy_coordination" not in information_set_mapping_to_gtlibrary[mapping]:
			return False

		index_gtlibrary = information_set_mapping_to_gtlibrary[mapping]["gtlibrary_index"]
		level, index_in_level = information_set_mapping_to_gtlibrary[mapping]["tensorcfr_strategy_coordination"]

		# convert a Numpy 1D array to a list of floats
		strategy = average_strategies[level][index_in_level].tolist()
		return_json[index_gtlibrary] = strategy

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	output_filename = "{}_{}.json".format(output_filename, datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))

	with open(os.path.join(output_dir, output_filename), 'w') as f:
		json.dump(return_json, f)

	return True