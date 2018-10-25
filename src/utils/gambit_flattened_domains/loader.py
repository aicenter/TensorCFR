#!/usr/bin/env python3
import copy
import os
import numpy as np
import hickle as hkl
from pprint import pprint

from src.commons import constants as common_constants
from src.utils.gambit_flattened_domains import constants
from src.utils.gambit_flattened_domains.parser import Parser


class TreeNode:
	def __init__(self, level=None, action_index=None):
		self.level = level
		self.action_index = action_index


class InformationSetManager:
	def __init__(self, level, number_of_information_sets):
		self.level = level
		self.number_of_information_sets = number_of_information_sets
		self.information_sets = {}
		self.information_set_acting_players_list = []

	def add_node(self, node):
		if node.type == constants.TERMINAL_NODE:
			return common_constants.INFOSET_FOR_TERMINAL_NODES

		if node.information_set_id not in self.information_sets:
			information_set_index = self.number_of_information_sets - 1 - len(self.information_sets)  # reverse infoset indices
			self.information_sets[node.information_set_id] = [information_set_index, node]
			self.information_set_acting_players_list.insert(0, node.information_set_id)
			return information_set_index
		else:
			return self.information_sets[node.information_set_id][0]

	def get_tensors(self, next_level_max_no_actions):
		information_set_acting_players = []
		initial_information_set_strategies = []

		for information_set_id in reversed(self.information_set_acting_players_list):
			node = self.information_sets[information_set_id][1]

			information_set_acting_players.insert(0, node.tensorcfr_id)

			if node.is_player():
				initial_player_infoset_strategy = [common_constants.IMAGINARY_PROBABILITIES] * next_level_max_no_actions

				actions = [float(1 / len(node.actions))] * len(node.actions)

				for index, action in enumerate(actions):
					initial_player_infoset_strategy[index] = action

				initial_information_set_strategies.insert(0, initial_player_infoset_strategy)
			elif node.is_chance():
				initial_chance_infoset_strategy = [np.nan] * next_level_max_no_actions

				for index, action in enumerate(reversed(node.actions)):
					initial_chance_infoset_strategy[index] = action['probability']

				initial_information_set_strategies.insert(0, initial_chance_infoset_strategy)
		return [
			np.asarray(information_set_acting_players, dtype=common_constants.INT_DTYPE_NUMPY),
			np.asarray(initial_information_set_strategies)
		]


class GambitLoader:

	def __init__(self, file, domain_name="from_gambit"):
		# if not os.path.isfile(file

		# check if there is a terminal node in any level
		self.__is_terminal_per_level = [False]
		# number of information sets per level
		self.__number_of_information_sets_per_level = [dict()]

		self.number_of_players = 2

		# domain name
		self.domain_name = domain_name
		# domain parameters
		self.domain_parameters = []

		self.actions_per_levels = []
		self.max_actions_per_levels = []

		# number of nodes per level starting from level 0
		self.nodes_per_levels = [1]  # level 0 has always one node
		self.number_of_levels = 0

		# the mapping of IS between `gtlibrary` and `TensorCFR` for computing best response
		self.information_set_strategy_index = 0
		self.information_set_mapping_to_gtlibrary = dict()

		# load meta information - number of levels, max actions per level...
		self.__load_meta_information(file)

		self.number_of_levels = len(self.nodes_per_levels)

		# init the list of utilities
		self.utilities = [None] * self.number_of_levels
		# init the list of node_to_infoset vectors
		self.node_to_infoset = [None] * self.number_of_levels
		# init a list of vectors with number of actions per node
		self.number_of_nodes_actions = [None] * self.number_of_levels

		self.initial_infoset_strategies = [None] * self.number_of_levels

		self.infoset_acting_players = [None] * self.number_of_levels

		for level, number_of_nodes in enumerate(self.nodes_per_levels):
			# set initial  utilities to zeros, will  be filled later
			self.utilities[level] = [common_constants.NON_TERMINAL_UTILITY] * number_of_nodes
			# set initial values for node_to_infoset
			self.node_to_infoset[level] = [None] * number_of_nodes
			# set initial values for zeros
			self.number_of_nodes_actions[level] = [0] * number_of_nodes

		self.__infoset_managers = [
			InformationSetManager(
					level=level,
					number_of_information_sets=self.__number_of_information_sets_per_level[level]
			)
			for level in range(len(self.actions_per_levels) + 1)
		]

		self.__generate_tensors(file)

	def __load_meta_information(self, file):
		lists_of_information_sets_ids_per_level = [dict()]
		# determines the maximum number of actions per level
		stack_nodes_lvl = [TreeNode(level=0)]

		with Parser(file) as parser:
			print(parser.header)
			self.domain_parameters = parser.header["domain_parameters"]

			for node in parser.next_node():
				tree_node = stack_nodes_lvl.pop()

				level = tree_node.level

				# the mapping of IS between `gtlibrary` and `TensorCFR` for computing best response
				if node.is_player() and \
						node.information_set_id not in self.information_set_mapping_to_gtlibrary.keys():
					self.information_set_mapping_to_gtlibrary[node.information_set_id] = {
						"gtlibrary_index": self.information_set_strategy_index,
						"tensorcfr_strategy_coordination": None
					}
					self.information_set_strategy_index += 1

				if not node.is_terminal():
					lists_of_information_sets_ids_per_level[level][node.information_set_id] = True

					if len(self.actions_per_levels) < (level + 1):
						self.actions_per_levels.append(0)
						self.max_actions_per_levels.append(0)
						self.__is_terminal_per_level.append(False)
						lists_of_information_sets_ids_per_level.append(dict())

					for _ in node.actions:
						new_level = level + 1
						stack_nodes_lvl.append(
								TreeNode(
										level=new_level,
								)
						)

					self.actions_per_levels[level] += len(node.actions)
					self.max_actions_per_levels[level] = max(len(node.actions), self.max_actions_per_levels[level])
				else:
					self.__is_terminal_per_level[level] = True
				self.number_of_levels = len(self.actions_per_levels)

			self.nodes_per_levels.extend(self.actions_per_levels)
			self.__number_of_information_sets_per_level = [len(information_sets) for information_sets in
			                                               lists_of_information_sets_ids_per_level]

	def __update_utilities(self, level, action_index, value):
		self.utilities[level][self.__placement_indices[level] + action_index] = value

	def __update_node_to_infoset(self, level, action_index, value):
		self.node_to_infoset[level][self.__placement_indices[level] + action_index] = value

	def __update_number_of_nodes_actions(self, level, action_index, value):
		self.number_of_nodes_actions[level][self.__placement_indices[level] + action_index] = value

	def __update_information_set_mapping_to_gtlibrary(self, node, level, node_to_infoset_value):
		# coordination == (level, index in the matrix)
		if node.is_player():
			self.information_set_mapping_to_gtlibrary[node.information_set_id]['tensorcfr_strategy_coordination'] = (
			level, node_to_infoset_value,)

	def __generate_tensors(self, file):
		# a vector of indices to filling vectors
		self.__placement_indices = copy.deepcopy(self.nodes_per_levels)
		self.__placement_indices[0] = 0
		# stack to safe nodes to visit, init with the root node
		nodes_stack = [TreeNode(level=0, action_index=0)]

		with Parser(file) as parser:
			for node in parser.next_node():
				tree_node = nodes_stack.pop()

				node_to_infoset_value = self.__infoset_managers[tree_node.level].add_node(node)
				self.__update_information_set_mapping_to_gtlibrary(node, tree_node.level, node_to_infoset_value)
				self.__update_node_to_infoset(tree_node.level, tree_node.action_index, node_to_infoset_value)

				if node.type != constants.TERMINAL_NODE:
					# count the number of actions of the current node
					actions_count = len(node.actions)
					# update the index of placement for the next level
					self.__placement_indices[tree_node.level + 1] -= actions_count

					self.__update_number_of_nodes_actions(tree_node.level, tree_node.action_index, actions_count)

					for action_index, action in enumerate(reversed(node.actions)):
						nodes_stack.append(TreeNode(level=tree_node.level + 1, action_index=action_index))
				else:
					# update utilities for a terminal node
					self.__update_utilities(tree_node.level, tree_node.action_index, node.payoffs[0])

		for level, max_number_of_actions in enumerate(self.max_actions_per_levels):
			[infoset_acting_players, initial_infoset_strategy] = self.__infoset_managers[level].get_tensors(
				max_number_of_actions)
			self.infoset_acting_players[level] = infoset_acting_players
			self.initial_infoset_strategies[level] = initial_infoset_strategy

	def show(self):
		print("domain_name: {}".format(self.domain_name))
		print("number_of_players: {}".format(self.number_of_players))
		print("number_of_levels: {}".format(self.number_of_levels))

		print("actions_per_levels:")
		pprint(self.actions_per_levels)
		print("max_actions_per_levels:")
		pprint(self.max_actions_per_levels)

		print("nodes_per_levels:")
		pprint(self.nodes_per_levels)
		print("node_to_infoset:")
		pprint(self.node_to_infoset, width=130)
		print("number_of_nodes_actions:")
		pprint(self.number_of_nodes_actions)
		print("utilities:")
		pprint(self.utilities, width=130)

		print("infoset_acting_players:")
		pprint(self.infoset_acting_players)
		print("initial_infoset_strategies:")
		pprint(self.initial_infoset_strategies)


class GambitLoaderCached(GambitLoader):
	def __init__(self, path, domain_name="from_gambit"):
		if not os.path.isfile(path):
			raise FileNotFoundError

		self.cached_dir_name = 'cached_hkl'
		self.cached_file_suffix = 'hkl'

		path_to_hkl = self._get_path_to_cached_hkl(path)

		if not os.path.isfile(path_to_hkl):
			super().__init__(path, domain_name)
			self._save_hkl(path_to_hkl)
		else:
			self._load_hkl(path_to_hkl)

	def _extract_clean_filename(self, path):
		""" docstring """
		last_pathname_item = path.split(os.sep)[-1]
		return last_pathname_item.split('.')[0]

	def _get_path_to_cached_hkl(self, path):
		gambit_file_m_time = os.path.getmtime(path)

		path_list = path.split(os.sep)
		path_list[0] = os.sep
		path_to_gambit_file_folder = path_list[:len(path_list) - 1]
		path_to_gambit_file_cached_folder = copy.deepcopy(path_to_gambit_file_folder)
		path_to_gambit_file_cached_folder.append(self.cached_dir_name)

		if not os.path.exists(os.path.join(*path_to_gambit_file_cached_folder)):
			os.makedirs(os.path.join(*path_to_gambit_file_cached_folder))

		filename_without_suffix = (path_list[-1]).split('.')[:-1]
		filename_npz = '{}__{}.{}'.format('.'.join(filename_without_suffix), str(gambit_file_m_time), self.cached_file_suffix)

		path_to_gambit_file_cached_folder.append(filename_npz)
		return os.path.join(*path_to_gambit_file_cached_folder)

	def _load_hkl(self, path_to_hkl):
		loaded = hkl.load(path_to_hkl)

		self.domain_name = loaded['domain_name']
		self.domain_parameters = loaded['domain_parameters']
		self.information_set_mapping_to_gtlibrary = loaded['information_set_mapping_to_gtlibrary']
		self.infoset_acting_players = loaded['infoset_acting_players']
		self.initial_infoset_strategies = loaded['initial_infoset_strategies']
		self.node_to_infoset = loaded['node_to_infoset']
		self.number_of_levels = loaded['number_of_levels']
		self.number_of_nodes_actions = loaded['number_of_nodes_actions']
		self.number_of_players = loaded['number_of_players']
		self.utilities = loaded['utilities']


	def _save_hkl(self, path_to_hkl):
		data = {
			'domain_name': self.domain_name,
			'domain_parameters': self.domain_parameters,
			'information_set_mapping_to_gtlibrary': self.information_set_mapping_to_gtlibrary,
			'infoset_acting_players': self.infoset_acting_players,
			'initial_infoset_strategies': self.initial_infoset_strategies,
			'node_to_infoset': self.node_to_infoset,
			'number_of_levels': self.number_of_levels,
			'number_of_nodes_actions': self.number_of_nodes_actions,
			'number_of_players': self.number_of_players,
			'utilities': self.utilities
		}

		hkl.dump(data, path_to_hkl, mode='w', compression='gzip')


if __name__ == '__main__':
	# domain01_efg = os.path.join(
	# 		common_constants.PROJECT_ROOT,
	# 		'doc',
	# 		'domain01_via_gambit.efg'
	# )
	# hunger_games_efg = os.path.join(
	# 	common_constants.PROJECT_ROOT,
	# 	'doc',
	# 	'hunger_games',
	# 	'hunger_games_via_gambit.efg'
	# )
	# hunger_games_2_efg = os.path.join(
	# 	common_constants.PROJECT_ROOT,
	# 	'doc',
	# 	'hunger_games_2',
	# 	'hunger_games_via_gambit.efg'
	# )
	# efg_files = [
	# 	domain01_efg,
	# 	hunger_games_efg,
	# 	hunger_games_2_efg
	# ]
	# domain_names = [
	# 	"domain01",
	# 	"hunger_games",
	# 	"hunger_games_2"
	# ]
	# for efg_file, domain_name in zip(efg_files, domain_names):
	# 	GambitLoader(efg_file, domain_name).show()
	# 	print("___________________________________\n")

	domain_name = os.path.join(
		common_constants.PROJECT_ROOT,
		'doc',
		'phantom_ttt',
		'SingleLevelPhantomTTT.gbt'
	)

	GambitLoader(domain_name).show()