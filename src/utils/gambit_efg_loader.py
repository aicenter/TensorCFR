import os
import re
import copy
import numpy as np
import tensorflow as tf

from src.commons import constants

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
	def __init__(self, level=None, coordinates=[]):
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

		self.flag_setted = False
		self.flag_imaginery_node_present = False
		self.flag_terminal_node_present = False

	def _is_imaginary_node_present(self, level, node_types):
		tensor = node_types[level]
		result = tensor[np.where(tensor == constants.IMAGINARY_NODE)]

		if result.shape == (0,):
			return False
		return True

	def _is_terminal_node_present(self, level, node_types):
		tensor = node_types[level]
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

	def make_infoset_acting_players(self, next_level_max_no_actions, node_types):
		if self.flag_setted == False and self.level > 0:
			self.flag_imaginery_node_present = self._is_imaginary_node_present(self.level, node_types)
			self.flag_setted = True

		infoset_acting_players = []
		current_infoset_strategies = []

		if self.flag_imaginery_node_present:
			self.infoset_dict['imaginary-node'] = [self.infoset_cnt, 'tnode', -1]  # last element - imaginery
			self.infoset_acting_players_list.append('imaginary-node')

		for idx, infoset_id in enumerate(self.infoset_acting_players_list):
			infoset_acting_players.append(self.infoset_dict[infoset_id][2])

			if self.infoset_dict[infoset_id][1] == constants.GAMBIT_NODE_TYPE_PLAYER:
				current_infoset_strategies.append(
					[float(1 / (next_level_max_no_actions * self.cnt_player_nodes))] * next_level_max_no_actions)
			elif self.infoset_dict[infoset_id][1] == constants.GAMBIT_NODE_TYPE_CHANCE:
				current_infoset_strategies.append(
						[action['probability'] for action in reversed(self.infoset_dict[infoset_id][3]['actions'])])
			else:
				current_infoset_strategies.append([0] * next_level_max_no_actions)

		return [infoset_acting_players, np.array(current_infoset_strategies)]

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

		self.actions_per_levels = []

		self.terminal_nodes_cnt = 0

		with open(efg_file) as self.gambit_file:
			game_header_line = self.gambit_file.readline()
			game_header = self.parse_gambit_header(game_header_line)

			if game_header['format'] == 'NFG':
				raise NotImplementedFormatException

			self.load()

		self.infoset_managers = [InformationSetManager(lvl) for lvl in range(len(self.actions_per_levels) + 1)]

		self.node_to_infoset = [None] * self.number_of_levels
		self.current_infoset_strategies = [None] * self.number_of_levels
		self.initial_infoset_strategies = [None] * self.number_of_levels  # TODO temporary because of tensorcfr.py
		self.infoset_acting_players = [None] * self.number_of_levels
		self.cumulative_regrets = [None] * (self.number_of_levels)
		self.positive_cumulative_regrets = [None] * (self.number_of_levels)

		self.node_types = [None] * (self.number_of_levels + 1)
		self.utilities = [None] * (self.number_of_levels + 1)
		self.node_to_infoset = [None] * (self.number_of_levels + 1)

		for idx in range(len(self.actions_per_levels)):
			self.infoset_acting_players[idx] = np.zeros(self.actions_per_levels[:idx]) * constants.NO_ACTING_PLAYER

		for idx in range(len(self.actions_per_levels) + 1):
			self.utilities[idx] = np.ones(self.actions_per_levels[:idx]) * constants.NON_TERMINAL_UTILITY
			self.node_types[idx] = np.ones(self.actions_per_levels[:idx]) * constants.IMAGINARY_NODE
			self.node_to_infoset[idx] = np.ones(self.actions_per_levels[:idx]) * TMP_NODE_TO_INFOSET_IMAGINARY * (-1)
		# self.cumulative_regrets[idx] = np.zeros(self.actions_per_levels[:idx])
		# self.positive_cumulative_regrets[idx] = np.zeros(self.actions_per_levels[:idx])

		with open(efg_file) as self.gambit_file:
			self.load_post()

	def parse_gambit_header(self, input_line):
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

	def parse_probability(self, probability_str):
		if '/' in probability_str:
			probability_list = probability_str.split('/')
			return float(int(probability_list[0]) / int(probability_list[1]))
		else:
			return float(probability_str)

	def parse_actions_chance(self, input_actions_str):
		parse_actions = re.findall(r'"(?P<name>[^"]*)" (?P<probability>[\d\./]+)', input_actions_str)
		return [{'name': action[0], 'probability': self.parse_probability(action[1])} for action in parse_actions]

	def parse_actions_player(self, input_actions_str):
		parse_actions = re.findall(r'"(?P<name>[^"]*)"', input_actions_str)
		return [{'name': action[0]} for action in parse_actions]

	def parse_payoffs(self, input_payoffs_str):
		parse_payoffs = re.findall(r'[\-]?[\d]+', input_payoffs_str)
		return [int(payoff) for payoff in parse_payoffs]

	def parse_chance_node(self, input_line):
		print(input_line)
		# r'^(?P<type>' + constants.GAMBIT_NODE_TYPE_CHANCE + ') "(?P<name>[^"]*)" (?P<information_set_number>\d+) "(?P<information_set_name>[^"]*)" \{(?P<actions_str>.*)\} (?P<outcome>\d+) "(?P<outcome_name>[^"]*)" \{(?P<payoffs_str>.*)\}',
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
		print(input_line)
		# r'^(?P<type>' + constants.GAMBIT_NODE_TYPE_PLAYER + ') "(?P<name>[^"]*)" (?P<player_number>\d+) (?P<information_set_number>\d+) "(?P<information_set_name>[^"]*)" \{(?P<actions_str>.*)\} (?P<outcome>\d+) "(?P<outcome_name>[^"]*)" \{(?P<payoffs_str>.*)\}',
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
		print(input_line)
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

	def is_gambit_node(self, line):
		if line[0] == constants.GAMBIT_NODE_TYPE_CHANCE or line[0] == constants.GAMBIT_NODE_TYPE_PLAYER or line[
			0] == constants.GAMBIT_NODE_TYPE_TERMINAL:
			return True
		return False

	def load(self):
		# determines the maximum number of actions per level
		stack_nodes_lvl = [TreeNode(level=0, coordinates=[])]

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
		stack_nodes_lvl = [TreeNode(level=0, coordinates=[])]

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

		for lvl in range(1, self.number_of_levels):
			self.node_to_infoset[lvl] = self.infoset_managers[lvl].make_node_to_infoset(self.node_to_infoset[lvl])

		for lvl in range(self.number_of_levels):
			[infoset_acting_players, infoset_strategies] = self.infoset_managers[lvl].make_infoset_acting_players(
					self.actions_per_levels[lvl], self.node_types)
			self.infoset_acting_players[lvl] = infoset_acting_players
			self.current_infoset_strategies[lvl] = infoset_strategies
			self.initial_infoset_strategies[lvl] = copy.deepcopy(self.current_infoset_strategies[lvl])
			self.cumulative_regrets[lvl] = np.zeros(infoset_strategies.shape)
			self.positive_cumulative_regrets[lvl] = np.zeros(infoset_strategies.shape)

		# self.infoset_acting_players[0] = self.infoset_acting_players[0][0]
		"""
		print('node_to_infoset lvl 0')
		print(self.node_to_infoset[0])
		print('infoset_acting_players lvl 1')
		print(self.node_to_infoset[1])
		print('infoset_acting_players lvl 2')
		print(self.node_to_infoset[2])

		print('node_to_infoset lvl 0')
		print(self.node_to_infoset[0])
		print('infoset_acting_players lvl 1')
		print(self.node_to_infoset[1])
		print('infoset_acting_players lvl 2')
		print(self.node_to_infoset[2])

		print('current_infoset_strategies lvl 0')
		print(self.current_infoset_strategies[0])
		print('current_infoset_strategies lvl 1')
		print(self.current_infoset_strategies[1])
		print('current_infoset_strategies lvl 2')
		print(self.current_infoset_strategies[2])

		print('infoset_acting_players lvl 0')
		print(self.infoset_acting_players[0])
		print('infoset_acting_players lvl 1')
		print(self.infoset_acting_players[1])
		print('infoset_acting_players lvl 2')
		print(self.infoset_acting_players[2])

		print('positive cumulative regrets lvl 0')
		print(self.positive_cumulative_regrets[0])
		print('positive cumulative regrets lvl 1')
		print(self.positive_cumulative_regrets[1])
		print('positive cumulative regrets lvl 2')
		print(self.positive_cumulative_regrets[2])
		"""

	def get_tensorflow_tensors(self):
		current_infoset_strategies = [None] * len(self.current_infoset_strategies)
		initial_infoset_strategies = [None] * len(self.initial_infoset_strategies)
		positive_cumulative_regrets = [None] * len(self.positive_cumulative_regrets)
		cumulative_regrets = [None] * len(self.cumulative_regrets)
		node_to_infoset = [None] * len(self.node_to_infoset)
		node_types = [None] * len(self.node_types)
		utilities = [None] * len(self.utilities)
		infoset_acting_players = [None] * len(self.infoset_acting_players)

		for lvl in range(self.number_of_levels):
			current_infoset_strategies[lvl] = tf.Variable(
					self.current_infoset_strategies[lvl],
					name='current_infoset_strategies_lvl{}'.format(lvl),
					dtype=tf.float32
			)
			initial_infoset_strategies[lvl] = tf.placeholder_with_default(
					self.initial_infoset_strategies[lvl],
					shape=self.initial_infoset_strategies[lvl].shape,
					name='initial_infoset_strategies_lvl{}'.format(lvl)
			)
			positive_cumulative_regrets[lvl] = tf.Variable(
					self.positive_cumulative_regrets[lvl],
					name='positive_cumulative_regrets_lvl{}'.format(lvl),
					dtype=tf.float32
			)
			cumulative_regrets[lvl] = tf.Variable(
					self.cumulative_regrets[lvl],
					name='cumulative_regrets_lvl{}'.format(lvl),
					dtype=tf.float32
			)
			node_to_infoset[lvl] = tf.Variable(
					self.node_to_infoset[lvl],
					name='node_to_infoset_lvl{}'.format(lvl),
					dtype=tf.int32
			)
			infoset_acting_players[lvl] = tf.Variable(
					self.infoset_acting_players[lvl],
					name='infoset_acting_players_lvl{}'.format(lvl),
					dtype=tf.int32
			)

		for lvl in range(self.number_of_levels + 1):
			node_types[lvl] = tf.Variable(self.node_types[lvl], name='node_types_lvl{}'.format(lvl), dtype=tf.int32)
			utilities[lvl] = tf.Variable(self.utilities[lvl], name='utilities_lvl{}'.format(lvl), dtype=tf.float32)

		return_dict = {
			'current_infoset_strategies' : current_infoset_strategies,
			'initial_infoset_strategies' : initial_infoset_strategies,
			'positive_cumulative_regrets': positive_cumulative_regrets,
			'node_to_infoset'            : node_to_infoset,
			'node_types'                 : node_types,
			'utilities'                  : utilities,
			'infoset_acting_players'     : infoset_acting_players
		}

		return return_dict


if __name__ == '__main__':
	domain01_efg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'doc', 'domain01_via_gambit.efg')
	mini_goofspiel_gbt = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'doc', 'mini_goofspiel',
	                                  'mini_goofspiel_via_gtlibrary.gbt')
	goofspiel_gbt = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'doc', 'goofspiel',
	                             'IIGS5_s1_bf_ft.gbt')
	phantomTTT = 'domain01_efg'

	domain = GambitEFGLoader(domain01_efg)

	domain_tensors = domain.get_tensorflow_tensors()

	current_infoset_strategies = domain_tensors['current_infoset_strategies']
	initial_infoset_strategies = domain_tensors['initial_infoset_strategies']
	positive_cumulative_regrets = domain_tensors['positive_cumulative_regrets']
	node_to_infoset = domain_tensors['node_to_infoset']
	node_types = domain_tensors['node_types']
	utilities = domain_tensors['utilities']
	infoset_acting_players = domain_tensors['infoset_acting_players']

	"""
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for lvl in range(domain.number_of_levels):
			print("Level " + str(lvl))
			print(sess.run(positive_cumulative_regrets[lvl]))
	"""
