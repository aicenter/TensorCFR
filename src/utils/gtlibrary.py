import datetime
import os
import json


def export_average_strategies_to_json(
		information_set_mapping_to_gtlibrary,
		average_strategies,
		output_filename,
		output_directory='out'):
	# TODO include game name and parameters
	output_filename = "{}_{}.json".format(output_filename, datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))

	return_json = list()

	for average_strategy_on_step in average_strategies:
		step = average_strategy_on_step['step']
		strategy = average_strategy_on_step['average_strategy']

		print(step)
		print(strategy)

		ret_strategy_tmp = [None] * len(information_set_mapping_to_gtlibrary)

		for mapping in information_set_mapping_to_gtlibrary:
			if 'gtlibrary_index' not in information_set_mapping_to_gtlibrary[mapping]:
				return False
			if 'tensorcfr_strategy_coordination' not in information_set_mapping_to_gtlibrary[mapping]:
				return False

			gtlib_idx = information_set_mapping_to_gtlibrary[mapping]["gtlibrary_index"]
			level, level_idx = information_set_mapping_to_gtlibrary[mapping]["tensorcfr_strategy_coordination"]

			# convert a Numpy 1D array to a list of floats
			ret_strategy = strategy[level][level_idx].tolist()
			ret_strategy_tmp[gtlib_idx] = ret_strategy

		return_json.append({"step": step, "strategy": ret_strategy_tmp})

	if not os.path.exists(output_directory):
		os.mkdir(output_directory)

	with open(os.path.join(output_directory, output_filename), 'w') as f:
		json.dump(return_json, f)

	return True