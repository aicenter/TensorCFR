import datetime
import os
import json


def export_average_strategies_to_json(
		domain,
		average_strategies,
		output_filename,
		output_directory='out'):
	output_filename = "{}_{}.json".format(output_filename, datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))

	information_set_mapping_to_gtlibrary = domain.information_set_mapping_to_gtlibrary

	return_json = list()
	return_json.append({"domain_name": domain.domain_name, "domain_parameters": domain.domain_parameters})

	for average_strategy_on_step in average_strategies:
		step = average_strategy_on_step['step']
		strategy = average_strategy_on_step['average_strategy']

		# returning strategy on this step
		return_strategy = [None] * len(information_set_mapping_to_gtlibrary)

		for mapping in information_set_mapping_to_gtlibrary:
			if 'gtlibrary_index' not in information_set_mapping_to_gtlibrary[mapping]:
				raise KeyError
			if 'tensorcfr_strategy_coordination' not in information_set_mapping_to_gtlibrary[mapping]:
				raise KeyError

			gtlib_idx = information_set_mapping_to_gtlibrary[mapping]["gtlibrary_index"]
			level, level_idx = information_set_mapping_to_gtlibrary[mapping]["tensorcfr_strategy_coordination"]

			# convert a Numpy 1D array to a list of floats
			return_strategy[gtlib_idx] = strategy[level][level_idx].tolist()

		return_json.append({"step": step, "strategy": return_strategy})

	if not os.path.exists(output_directory):
		os.mkdir(output_directory)

	with open(os.path.join(output_directory, output_filename), 'w') as f:
		json.dump(return_json, f)
