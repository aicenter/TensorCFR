#!/usr/bin/env python3

import os
import re
import copy
import numpy as np
import tensorflow as tf

from src.commons import constants

# TODO @janrudolf: Why are there these 2 constants?
# TODO @janrudolf: And why are they set to 7 and 8? It should be the last and the penultimate.
TMP_NODE_TO_INFOSET_TERMINAL = 7
TMP_NODE_TO_INFOSET_IMAGINARY = 8


class NotAcceptableFormatException:
	pass


class NotRecognizedPlayersException:
	pass


class NotRecognizedTreeNodeException:
	pass


class NotImplementedFormatException:
	pass


class TreeNode:
	def __init__(self, level=None, coordinates=None):
		if coordinates is None:
			coordinates = []
		self.level = level
		self.coordinates = coordinates


class InformationSetManager:
	def __init__(self, level):
		self.level = level
		self.infoset_cnt = 0
		self.infoset_dict = {}
		self.infoset_node_to_infoset = []
		self.infoset_acting_players_list = []

		self.cnt_player_nodes = 0

		self.flag_set = False
		self.flag_imaginary_node_present = False
		self.flag_terminal_node_present = False

	@staticmethod
	def _is_imaginary_node_present(level, node_types_):
		tensor = node_types_[level]
		result = tensor[np.where(tensor == constants.IMAGINARY_NODE)]

		if result.shape == (0,):
			return False
		return True

	@staticmethod
	def _is_terminal_node_present(level, node_types_):
		tensor = node_types_[level]
		result = tensor[np.where(tensor == constants.TERMINAL_NODE)]

		if result.shape == (0,):
			return False
		return True

	def add(self, node):
		# node type due to _is_imaginary_node_present
		self.infoset_node_to_infoset.append(node['infoset_id'])

		if node['type'] == constants.GAMBIT_NODE_TYPE_PLAYER:
			self.cnt_player_nodes += 1

		if node['infoset_id'] not in self.infoset_dict:
			return_node_to_infoset_value = self.infoset_cnt
			self.infoset_dict[node['infoset_id']] = [return_node_to_infoset_value, node['type'], node['tensorcfr_id'], node]
			self.infoset_acting_players_list.insert(0, node['infoset_id'])
			self.infoset_cnt += 1
			return return_node_to_infoset_value
		else:
			return_node_to_infoset_value = self.infoset_dict[node['infoset_id']][0]
			return return_node_to_infoset_value

	def make_infoset_acting_players(self, next_level_max_no_actions, node_types_):
		if self.flag_set is False and self.level > 0:
			self.flag_imaginary_node_present = self._is_imaginary_node_present(self.level, node_types_)
			self.flag_set = True

		infoset_acting_players_ = []
		current_infoset_strategies_ = []

		if self.flag_imaginary_node_present:
			self.infoset_dict['imaginary-node'] = [self.infoset_cnt, 'tnode', -1]  # last element - imaginary
			self.infoset_acting_players_list.append('imaginary-node')

		for idx, infoset_id in enumerate(self.infoset_acting_players_list):
			infoset_acting_players_.append(self.infoset_dict[infoset_id][2])

			if self.infoset_dict[infoset_id][1] == constants.GAMBIT_NODE_TYPE_PLAYER:
				# TODO: @janrudolf Fix here
				#  This is not the correct way to compute uniform strategies. Normalization is not over all nodes in the next
				#  level. It is over number of children from the information set (alternatively number of children at each node
				#  of the information set). But one also needs to keep in mind imaginary node which should not be counted
				#  towards the normalization sum.
				#
				#  Check out the method `get_infoset_uniform_strategies()` located at line 266 of
				#  `algorithms.tensorcfr.TensorCFR.py`.
				current_infoset_strategies_.append(
					# [float(1 / (next_level_max_no_actions * self.cnt_player_nodes))] * next_level_max_no_actions
					[np.nan] * next_level_max_no_actions    # TODO This is a hotfix.
				)
			elif self.infoset_dict[infoset_id][1] == constants.GAMBIT_NODE_TYPE_CHANCE:
				current_infoset_strategy = [np.nan] * next_level_max_no_actions

				for index, action in enumerate(reversed(self.infoset_dict[infoset_id][3]['actions'])):
					current_infoset_strategy[index] = action['probability']

				current_infoset_strategies_.append(current_infoset_strategy)
			else:
				# current_infoset_strategies_.append([0] * next_level_max_no_actions)
				# TODO Just to be sure, let's put NaNs everywhere.
				current_infoset_strategies_.append([np.nan] * next_level_max_no_actions)

		return [np.asarray(infoset_acting_players_), np.asarray(current_infoset_strategies_)]

	def make_node_to_infoset(self, tensor):
		tensor[np.where(tensor == ((-1) * TMP_NODE_TO_INFOSET_IMAGINARY))] = -self.infoset_cnt
		tensor[np.where(tensor >= 0)] += -(self.infoset_cnt - 1)
		tensor[np.where(tensor != 0)] *= -1
		return tensor


class GambitEFGLoader:

	def __init__(self, efg_file):
		self.gambit_filename = efg_file
		self.nodes = list()
		self.number_of_players = 2

		self.domain_name = ""
		self.actions_per_levels = []
		self.number_of_levels = 0

		self.terminal_nodes_cnt = 0

		with open(efg_file) as self.gambit_file:
			game_header_line = self.gambit_file.readline()
			game_header = self.parse_gambit_header(game_header_line)

			if game_header['format'] == 'NFG':
				raise NotImplementedFormatException

			self.domain_name = game_header['name']

			self.load()

		self.infoset_managers = [InformationSetManager(lvl) for lvl in range(len(self.actions_per_levels) + 1)]

		self.node_to_infoset = [None] * self.number_of_levels
		self.current_infoset_strategies = [None] * self.number_of_levels
		self.initial_infoset_strategies = [None] * self.number_of_levels  # TODO temporary because of TensorCFR.py
		self.infoset_acting_players = [None] * self.number_of_levels
		self.cumulative_regrets = [None] * self.number_of_levels
		self.positive_cumulative_regrets = [None] * self.number_of_levels

		self.node_types = [None] * (self.number_of_levels + 1)
		self.utilities = [None] * (self.number_of_levels + 1)
		self.node_to_infoset = [None] * (self.number_of_levels + 1)

		for idx in range(len(self.actions_per_levels) + 1):
			self.utilities[idx] = np.ones(self.actions_per_levels[:idx]) * constants.NON_TERMINAL_UTILITY
			self.node_types[idx] = np.ones(self.actions_per_levels[:idx], dtype=np.int) * constants.IMAGINARY_NODE
			self.node_to_infoset[idx] = - np.ones(self.actions_per_levels[:idx], dtype=np.int) * TMP_NODE_TO_INFOSET_IMAGINARY

		with open(efg_file) as self.gambit_file:
			self.load_post()

	@staticmethod
	def parse_gambit_header(input_line):
		results = re.search(r'^(?P<format>EFG|NFG) (?P<version>\d) R "(?P<name>[^"]+)" \{(?P<players_dirty>.*)\}',
		                    input_line)
		if results:
			results_players = re.findall(r'"([^"]+)"', results.group('players_dirty'))
			if results_players is None:
				raise NotRecognizedPlayersException
			return_dict = {
				'format' : results.group('format'),
				'version': int(results.group('version')),
				'name'   : results.group('name'),
				'players': results_players
			}
			return return_dict
		else:
			raise NotAcceptableFormatException

	def parse_node(self, input_line):
		# http://www.gambit-project.org/gambit13/formats.html
		if len(input_line) == 0:
			return False

		node_type = input_line[0]

		if node_type == constants.GAMBIT_NODE_TYPE_CHANCE:
			return self.parse_chance_node(input_line)
		elif node_type == constants.GAMBIT_NODE_TYPE_PLAYER:
			return self.parse_player_node(input_line)
		elif node_type == constants.GAMBIT_NODE_TYPE_TERMINAL:
			return self.parse_terminal_node(input_line)
		else:
			return False

	@staticmethod
	def parse_probability(probability_str):
		if '/' in probability_str:
			probability_list = probability_str.split('/')
			return float(int(probability_list[0]) / int(probability_list[1]))
		else:
			return float(probability_str)

	def parse_actions_chance(self, input_actions_str):
		parse_actions = re.findall(r'"(?P<name>[^"]*)" (?P<probability>[\d\./]+)', input_actions_str)
		return [{'name': action[0], 'probability': self.parse_probability(action[1])} for action in parse_actions]

	@staticmethod
	def parse_actions_player(input_actions_str):
		parse_actions = re.findall(r'"(?P<name>[^"]*)"', input_actions_str)
		return [{'name': action[0]} for action in parse_actions]

	@staticmethod
	def parse_payoffs(input_payoffs_str):
		parse_payoffs = re.findall(r'[\-]?[\d]+', input_payoffs_str)
		return [int(payoff) for payoff in parse_payoffs]

	def parse_chance_node(self, input_line):
		parse_line = re.search(
				r'^(?P<type>' + constants.GAMBIT_NODE_TYPE_CHANCE + ') "(?P<name>[^"]*)" (?P<information_set_number>\d+)\ ?"?(?P<information_set_name_optional>[^"]*)"?\ ?\{?(?P<actions_optional>[^\}]*)\}?\ ?(?P<outcome>\d+)\ ?"?(?P<outcome_name_optional>[^"]*)"?\ ?\{?(?P<payoffs_optional>.*)\}?',
				input_line
		)

		actions = self.parse_actions_chance(parse_line.group('actions_optional'))
		payoffs = self.parse_payoffs(parse_line.group('payoffs_optional'))
		infoset_id = 'c-' + parse_line.group('information_set_number')

		return {
			'type'                  : parse_line.group('type'),
			'name'                  : parse_line.group('name'),
			'information_set_number': int(parse_line.group('information_set_number')),
			'information_set_name'  : parse_line.group('information_set_name_optional'),
			'actions'               : actions,
			'outcome'               : parse_line.group('outcome'),
			'outcome_name'          : parse_line.group('outcome_name_optional'),
			'payoffs'               : payoffs,
			'tensorcfr_id'          : constants.CHANCE_PLAYER,
			'infoset_id'            : infoset_id
		}

	def parse_player_node(self, input_line):
		parse_line = re.search(
				r'^(?P<type>' + constants.GAMBIT_NODE_TYPE_PLAYER + ') "(?P<name>[^"]*)" (?P<player_number>\d+) (?P<information_set_number>\d+)\ ?"?(?P<information_set_name>[^"]*)"?\ ?\{?(?P<actions_optional>[^\}]*)\}?\ ?(?P<outcome>\d+)\ ?"?(?P<outcome_name>[^"]*)"?\ ?\{?(?P<payoffs_optional>[^\}]*)\}?',
				input_line
		)

		actions = self.parse_actions_player(parse_line.group('actions_optional'))
		payoffs = self.parse_payoffs(parse_line.group('payoffs_optional'))
		infoset_id = 'p-' + parse_line.group('player_number') + '-' + parse_line.group('information_set_number')

		return {
			'type'                  : parse_line.group('type'),
			'name'                  : parse_line.group('name'),
			'player_number'         : int(parse_line.group('player_number')),
			'information_set_number': int(parse_line.group('information_set_number')),
			'information_set_name'  : parse_line.group('information_set_name'),
			'actions'               : actions,
			'outcome'               : parse_line.group('outcome'),
			'outcome_name'          : parse_line.group('outcome_name'),
			'payoffs'               : payoffs,
			'tensorcfr_id'          : int(parse_line.group('player_number')),
			'infoset_id'            : infoset_id
		}

	def parse_terminal_node(self, input_line):
		parse_line = re.search(
				r'^(?P<type>' + constants.GAMBIT_NODE_TYPE_TERMINAL + ') "(?P<name>[^"]*)" (?P<outcome>\d+)\ ?"?(?P<outcome_name_optional>[^"]*)"?\ ?\{(?P<payoffs>.*)\}',
				input_line
		)

		payoffs = self.parse_payoffs(parse_line.group('payoffs'))
		# infoset_id = 't-' + str(self.terminal_nodes_cnt)
		infoset_id = 't'

		self.terminal_nodes_cnt += 1

		return {
			'type'        : parse_line.group('type'),
			'name'        : parse_line.group('name'),
			'outcome'     : parse_line.group('outcome'),
			'outcome_name': parse_line.group('outcome_name_optional'),
			'payoffs'     : payoffs,
			'tensorcfr_id': constants.NO_ACTING_PLAYER,
			'infoset_id'  : infoset_id
		}

	@staticmethod
	def is_gambit_node(line):
		if line[0] == constants.GAMBIT_NODE_TYPE_CHANCE or line[0] == constants.GAMBIT_NODE_TYPE_PLAYER or line[
			0] == constants.GAMBIT_NODE_TYPE_TERMINAL:
			return True
		return False

	def load(self):
		# determines the maximum number of actions per level
		stack_nodes_lvl = [TreeNode(level=0)]

		for cnt, line in enumerate(self.gambit_file):
			if self.is_gambit_node(line):
				node = self.parse_node(line)

				tree_node = stack_nodes_lvl.pop()

				level = tree_node.level
				coordinates = tree_node.coordinates

				if node['type'] != constants.GAMBIT_NODE_TYPE_TERMINAL:
					if len(self.actions_per_levels) < (level + 1):
						self.actions_per_levels.append(0)

					for idx, action in enumerate(reversed(node['actions'])):
						new_level = level + 1
						new_coordinates = copy.deepcopy(coordinates)
						new_coordinates.append(idx)
						stack_nodes_lvl.append(TreeNode(level=new_level, coordinates=new_coordinates))

					self.actions_per_levels[level] = max(len(node['actions']), self.actions_per_levels[level])
			self.number_of_levels = len(self.actions_per_levels)

	def update_utilities(self, level, coordinates, value):
		if level == 0:
			self.utilities[level] = value
		else:
			self.utilities[level][tuple(coordinates)] = value

	def update_node_types(self, level, coordinates, value):
		if level == 0:
			self.node_types[level] = value
		else:
			self.node_types[level][tuple(coordinates)] = value

	def update_node_to_infoset(self, level, coordinates, value):
		if level == 0:
			self.node_to_infoset[level] = value
		else:
			self.node_to_infoset[level][tuple(coordinates)] = value

	def load_post(self):
		stack_nodes_lvl = [TreeNode(level=0)]

		for cnt, line in enumerate(self.gambit_file):
			if self.is_gambit_node(line):
				node = self.parse_node(line)

				tree_node = stack_nodes_lvl.pop()

				level = tree_node.level
				coordinates = tree_node.coordinates

				node_to_infoset_value = self.infoset_managers[level].add(node)
				self.update_node_to_infoset(level, coordinates, node_to_infoset_value)

				if node['type'] == constants.GAMBIT_NODE_TYPE_CHANCE or node['type'] == constants.GAMBIT_NODE_TYPE_PLAYER:
					self.update_node_types(level, coordinates, constants.INNER_NODE)

					for idx, action in enumerate(reversed(node['actions'])):
						new_level = level + 1
						new_coordinates = copy.deepcopy(coordinates)
						new_coordinates.append(idx)
						stack_nodes_lvl.append(TreeNode(level=new_level, coordinates=new_coordinates))
				else:
					self.update_utilities(level, coordinates, node['payoffs'][0])
					self.update_node_types(level, coordinates, constants.TERMINAL_NODE)

		for level in range(1, self.number_of_levels):
			self.node_to_infoset[level] = self.infoset_managers[level].make_node_to_infoset(self.node_to_infoset[level])

		for level in range(self.number_of_levels):
			[infoset_acting_players_, infoset_strategies] = self.infoset_managers[level].make_infoset_acting_players(
					self.actions_per_levels[level], self.node_types)
			self.infoset_acting_players[level] = infoset_acting_players_
			self.current_infoset_strategies[level] = infoset_strategies
			self.initial_infoset_strategies[level] = np.array(copy.deepcopy(self.current_infoset_strategies[level]))
			self.cumulative_regrets[level] = np.zeros(infoset_strategies.shape)
			self.positive_cumulative_regrets[level] = np.zeros(infoset_strategies.shape)

	def get_tensorflow_tensors(self):
		current_infoset_strategies_ = [None] * len(self.current_infoset_strategies)
		initial_infoset_strategies_ = [None] * len(self.initial_infoset_strategies)
		positive_cumulative_regrets_ = [None] * len(self.positive_cumulative_regrets)
		cumulative_regrets = [None] * len(self.cumulative_regrets)
		node_to_infoset_ = [None] * len(self.node_to_infoset)
		node_types_ = [None] * len(self.node_types)
		utilities_ = [None] * len(self.utilities)
		infoset_acting_players_ = [None] * len(self.infoset_acting_players)

		for level in range(self.number_of_levels):
			current_infoset_strategies_[level] = tf.Variable(
					self.current_infoset_strategies[level],
					name='current_infoset_strategies_lvl{}'.format(level),
					dtype=tf.float32
			)
			initial_infoset_strategies_[level] = tf.placeholder_with_default(
					self.initial_infoset_strategies[level],
					shape=self.initial_infoset_strategies[level].shape,
					name='initial_infoset_strategies_lvl{}'.format(level)
			)
			positive_cumulative_regrets_[level] = tf.Variable(
					self.positive_cumulative_regrets[level],
					name='positive_cumulative_regrets_lvl{}'.format(level),
					dtype=tf.float32
			)
			cumulative_regrets[level] = tf.Variable(
					self.cumulative_regrets[level],
					name='cumulative_regrets_lvl{}'.format(level),
					dtype=tf.float32
			)
			node_to_infoset_[level] = tf.Variable(
					self.node_to_infoset[level],
					name='node_to_infoset_lvl{}'.format(level),
					dtype=tf.int32
			)
			infoset_acting_players_[level] = tf.Variable(
					self.infoset_acting_players[level],
					name='infoset_acting_players_lvl{}'.format(level),
					dtype=tf.int32
			)

		for level in range(self.number_of_levels + 1):
			node_types_[level] = tf.Variable(self.node_types[level], name='node_types_lvl{}'.format(level), dtype=tf.int32)
			utilities_[level] = tf.Variable(self.utilities[level], name='utilities_lvl{}'.format(level), dtype=tf.float32)

		return_dict = {
			'current_infoset_strategies' : current_infoset_strategies_,
			'initial_infoset_strategies' : initial_infoset_strategies_,
			'positive_cumulative_regrets': positive_cumulative_regrets_,
			'node_to_infoset'            : node_to_infoset_,
			'node_types'                 : node_types_,
			'utilities'                  : utilities_,
			'infoset_acting_players'     : infoset_acting_players_
		}

		return return_dict


if __name__ == '__main__':
	domain01_efg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'doc', 'domain01_via_gambit.efg')
	mini_goofspiel_gbt = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'doc', 'mini_goofspiel',
	                                  'mini_goofspiel_via_gtlibrary.gbt')
	# noinspection SpellCheckingInspection
	goofspiel_gbt = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'doc', 'goofspiel',
	                             'IIGS5_s1_bf_ft.gbt')

	"""
	poker = os.path.join(
		os.path.dirname(
			os.path.abspath(
				__file__)
		),
		'..',
		'..',
		'doc',
		'poker',
		'GP_cards2x2_122.gbt'
	)
	"""

	domain = GambitEFGLoader(domain01_efg)

	for level in range(len(domain.actions_per_levels)):
		print(domain.current_infoset_strategies[level])
