import hickle


def load_average_strategies(filename):
	with open(filename) as f:
		return hickle.load(f)
	return None

def store_average_strategies(filename, average_strategies):
	stored_list = []

	for average_strategy in average_strategies:
		stored_list.append({'step': average_strategy['step'], 'average_strategy': average_strategy['average_strategy']})

	with open(filename, 'w') as f:
		hickle.dump(stored_list, f, mode='w', compression='gzip')