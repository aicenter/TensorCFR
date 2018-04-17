import os
import re
import copy
import numpy as np
import tensorflow as tf

from src.commons import constants

TMP_NODE_TO_INFOSET_TERMINAL = 7
TMP_NODE_TO_INFOSET_IMAGINERY = 8


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
		self.infoset_acting_player_list = []

		self.cnt_player_nodes = 0

		self.flag_setted = False
		self.flag_imaginery_node_present = False

	def _set_node_to_infoset(self, coordinates, level, node_to_infoset, value):
		if level:
			node_to_infoset[level][tuple(coordinates)] = value
		else:
			node_to_infoset[level] = value
		return True

	def _is_imaginery_node_present(self, level, node_types):
		tensor = node_types[level]
		result = tensor[np.where(tensor == constants.IMAGINARY_NODE)]

		if result.shape == (0,):
			return False
		return True


	def add(self, node, coordinates, level, node_to_infoset):
		# node type kvuli _is_imaginery_node_present
		self.infoset_node_to_infoset.append(node['infoset_id'])

		if node['type'] == constants.GAMBIT_NODE_TYPE_PLAYER:
			self.cnt_player_nodes += 1

		# set node to infoset value
		if node['infoset_id'] not in self.infoset_dict:
			self._set_node_to_infoset(coordinates, level, node_to_infoset, self.infoset_cnt)
		else:
			self._set_node_to_infoset(coordinates, level, node_to_infoset, self.infoset_dict[node['infoset_id']][0])

		if node['infoset_id'] not in self.infoset_dict:
			self.infoset_dict[node['infoset_id']] = [self.infoset_cnt, node['type'], node['tensorcfr_id'], node]
			self.infoset_acting_player_list.insert(0, node['infoset_id'])
			self.infoset_cnt += 1
			return True

		return False

	def make_infoset_acting_player(self, next_level_max_no_actions, node_types):
		if self.flag_setted == False and self.level > 0:
			self.flag_imaginery_node_present = self._is_imaginery_node_present(self.level, node_types)
			self.flag_setted = True

		infoset_acting_player = []
		current_infoset_strategies = []

		if self.flag_imaginery_node_present:
			print("Pridavam imag v levelu " + str(self.level))
			self.infoset_dict['imaginary-node'] = [self.infoset_cnt, 'tnode', -1] #last element - imaginery
			self.infoset_acting_player_list.append('imaginary-node')

		for idx, infoset_id in enumerate(self.infoset_acting_player_list):
			infoset_acting_player.append(self.infoset_dict[infoset_id][2])

			if self.infoset_dict[infoset_id][1] == constants.GAMBIT_NODE_TYPE_PLAYER:
				current_infoset_strategies.append([float(1/(next_level_max_no_actions*self.cnt_player_nodes))] * next_level_max_no_actions)
			elif self.infoset_dict[infoset_id][1] == constants.GAMBIT_NODE_TYPE_CHANCE:
				current_infoset_strategies.append([action['probability'] for action in reversed(self.infoset_dict[infoset_id][3]['actions'])])
			else:
				current_infoset_strategies.append([0] * next_level_max_no_actions)

		return [infoset_acting_player, np.array(current_infoset_strategies)]

	def make_node_to_infoset(self, tensor):
		tensor[np.where(tensor == ((-1)*TMP_NODE_TO_INFOSET_IMAGINERY))] = -self.infoset_cnt
		tensor[np.where(tensor >= 0)] += -(self.infoset_cnt - 1)
		tensor[np.where(tensor != 0)] *= -1
		return tensor

class GambitEFGLoader:

	def __init__(self, efg_file):
		self.gambit_filename = efg_file
		self.nodes = list()
		self.number_of_players = 2

		self.max_actions_per_level = []

		self.terminal_nodes_cnt = 0

		with open(efg_file) as self.gambit_file:
			self.load()

		with open(efg_file) as self.gambit_file:
			self.load_post()

	def parse_node(self, input_line):
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

	def parse_actions_chance(self, input_actions_str):
		parse_actions = re.findall(r'"(?P<name>[^"]*)" (?P<probability>[\d\.]+)', input_actions_str)
		return [{'name': action[0], 'probability': float(action[1])} for action in parse_actions]

	def parse_actions_player(self, input_actions_str):
		parse_actions = re.findall(r'"(?P<name>[^"]*)"', input_actions_str)
		return [{'name': action[0]} for action in parse_actions]

	def parse_payoffs(self, input_payoffs_str):
		parse_payoffs = re.findall(r'[\-]?[\d]+', input_payoffs_str)
		return [int(payoff) for payoff in parse_payoffs]

	def parse_chance_node(self, input_line):
		parse_line = re.search(
			r'^(?P<type>' + constants.GAMBIT_NODE_TYPE_CHANCE + ') "(?P<name>[^"]*)" (?P<information_set_number>\d+) "(?P<information_set_name>[^"]*)" \{ (?P<actions_str>.*) \} (?P<outcome>\d+) "(?P<outcome_name>[^"]*)" \{ (?P<payoffs_str>.*) \}',
			input_line
		)

		actions = self.parse_actions_chance(parse_line.group('actions_str'))
		payoffs = self.parse_payoffs(parse_line.group('payoffs_str'))
		infoset_id = 'c-' + parse_line.group('information_set_number')

		return {
			'type': parse_line.group('type'),
			'name': parse_line.group('name'),
			'information_set_number': int(parse_line.group('information_set_number')),
			'information_set_name': parse_line.group('information_set_name'),
			'actions': actions,
			'outcome': parse_line.group('outcome'),
			'outcome_name': parse_line.group('outcome_name'),
			'payoffs': payoffs,
			'tensorcfr_id': constants.CHANCE_PLAYER,
			'infoset_id': infoset_id
		}

	def parse_player_node(self, input_line):
		parse_line = re.search(
			r'^(?P<type>' + constants.GAMBIT_NODE_TYPE_PLAYER + ') "(?P<name>[^"]*)" (?P<player_number>\d+) (?P<information_set_number>\d+) "(?P<information_set_name>[^"]*)" \{ (?P<actions_str>.*) \} (?P<outcome>\d+) "(?P<outcome_name>[^"]*)" \{ (?P<payoffs_str>.*) \}',
			input_line
		)

		actions = self.parse_actions_player(parse_line.group('actions_str'))
		payoffs = self.parse_payoffs(parse_line.group('payoffs_str'))
		infoset_id = 'p-' + parse_line.group('player_number') + '-' + parse_line.group('information_set_number')

		return {
			'type': parse_line.group('type'),
			'name': parse_line.group('name'),
			'player_number': int(parse_line.group('player_number')),
			'information_set_number': int(parse_line.group('information_set_number')),
			'information_set_name': parse_line.group('information_set_name'),
			'actions': actions,
			'outcome': parse_line.group('outcome'),
			'outcome_name': parse_line.group('outcome_name'),
			'payoffs': payoffs,
			'tensorcfr_id': int(parse_line.group('player_number')),
			'infoset_id': infoset_id
		}

	def parse_terminal_node(self, input_line):
		parse_line = re.search(
			r'^(?P<type>' + constants.GAMBIT_NODE_TYPE_TERMINAL + ') "(?P<name>[^"]*)" (?P<outcome>\d+) "(?P<outcome_name>[^"]*)" \{ (?P<payoffs_str>.*) \}',
			input_line
		)

		payoffs = self.parse_payoffs(parse_line.group('payoffs_str'))
		infoset_id = 't-' + str(self.terminal_nodes_cnt)
		#infoset_id = 't'

		self.terminal_nodes_cnt += 1

		return {
			'type': parse_line.group('type'),
			'name': parse_line.group('name'),
			'outcome': parse_line.group('outcome'),
			'outcome_name': parse_line.group('outcome_name'),
			'payoffs': payoffs,
			'tensorcfr_id': constants.NO_ACTING_PLAYER,
			'infoset_id': infoset_id
		}

	def load(self):
		# max actions per level
		stack_nodes_lvl = [TreeNode(level=0, coordinates=[])]

		cnt = 1
		for line in self.gambit_file:
			# in domain01 nodes starts at line 4
			if cnt >= 4:
				node = self.parse_node(line)

				tree_node = stack_nodes_lvl.pop()

				level = tree_node.level
				coordinates = tree_node.coordinates

				#print(node['type'], 'level {}'.format(level), self.max_actions_per_level)

				# actions per level
				if node['type'] != constants.GAMBIT_NODE_TYPE_TERMINAL:
					if len(self.max_actions_per_level) < (level + 1):
						self.max_actions_per_level.append(0)

					#print(node['actions'], coordinates)

					for index, action in enumerate(reversed(node['actions'])):
						#print(" - add {} {}".format(action['name'], level + 1))
						new_level = level + 1
						new_coordinates = copy.deepcopy(coordinates)
						new_coordinates.append(index)
						stack_nodes_lvl.append(TreeNode(level=new_level, coordinates=new_coordinates))

					self.max_actions_per_level[level] = max(len(node['actions']), self.max_actions_per_level[level])
			cnt += 1

		#print("Po for cyklu:")
		#print(stack_nodes_lvl)

		#print(self.max_actions_per_level)

	def load_post(self):
		stack_nodes_lvl = [TreeNode(level=0, coordinates=[])]

		self.node_type = [None] * (len(self.max_actions_per_level) + 1)
		self.infoset_acting_player = [None] * (len(self.max_actions_per_level) + 1)
		#create positive cumulative regrets
		self.utilities = [None] * (len(self.max_actions_per_level) + 1)
		self.node_to_infoset = [None] * (len(self.max_actions_per_level) + 1)
		self.cumulative_regrets = [None] * (len(self.max_actions_per_level) + 1)
		self.positive_cumulative_regrets = [None] * (len(self.max_actions_per_level) + 1)

		infoset_managers = [ InformationSetManager(lvl) for lvl in range(len(self.max_actions_per_level) + 1)]

		for idx in range(len(self.max_actions_per_level) + 1):
			self.utilities[idx] = np.ones(self.max_actions_per_level[:idx]) * constants.NON_TERMINAL_UTILITY
			self.node_type[idx] = np.ones(self.max_actions_per_level[:idx]) * constants.IMAGINARY_NODE
			self.infoset_acting_player[idx] = np.zeros(self.max_actions_per_level[:idx]) * constants.NO_ACTING_PLAYER
			self.node_to_infoset[idx] = np.ones(self.max_actions_per_level[:idx]) * TMP_NODE_TO_INFOSET_IMAGINERY * (-1)
			self.cumulative_regrets[idx] = np.zeros(self.max_actions_per_level[:idx])
			self.positive_cumulative_regrets[idx] = np.zeros(self.max_actions_per_level[:idx])

		# walk the tree
		cnt = 1
		for line in self.gambit_file:
			if cnt >= 4:
				node = self.parse_node(line)

				tree_node = stack_nodes_lvl.pop()

				level = tree_node.level
				coordinates = tree_node.coordinates

				infoset_managers[level].add(node, coordinates, level, self.node_to_infoset)

				if node['type'] != constants.GAMBIT_NODE_TYPE_TERMINAL:
					if level > 0:
						self.node_type[level][tuple(coordinates)] = constants.INNER_NODE
					else:
						self.node_type[level] = constants.INNER_NODE


					for index, action in enumerate(reversed(node['actions'])):
						new_level = level + 1
						new_coordinates = copy.deepcopy(coordinates)
						new_coordinates.append(index)
						stack_nodes_lvl.append(TreeNode(level=new_level, coordinates=new_coordinates))
				else:
					if level > 0:
						self.utilities[level][tuple(coordinates)] = node['payoffs'][0]
						self.node_type[level][tuple(coordinates)] = constants.TERMINAL_NODE
					else:
						self.utilities[level] = node['payoffs'][0]
						self.node_type[level] = constants.TERMINAL_NODE

				if level == 0:
					self.infoset_acting_player[level] = constants.NO_ACTING_PLAYER
			cnt += 1

		for lvl in range(1,3):
			self.node_to_infoset[lvl] = infoset_managers[lvl].make_node_to_infoset(self.node_to_infoset[lvl])

		[infoset_acting_player_lvl0, infoset_strategies_lvl0] = infoset_managers[0].make_infoset_acting_player(5, self.node_type)
		[infoset_acting_player_lvl1, infoset_strategies_lvl1] = infoset_managers[1].make_infoset_acting_player(3, self.node_type)
		[infoset_acting_player_lvl2, infoset_strategies_lvl2] = infoset_managers[2].make_infoset_acting_player(2, self.node_type)

		self.infoset_acting_player = [infoset_acting_player_lvl0, infoset_acting_player_lvl1, infoset_acting_player_lvl2]
		self.current_infoset_strategies = [infoset_strategies_lvl0, infoset_strategies_lvl1, infoset_strategies_lvl2]

		print("current_infoset_strategies")
		print('lvl 0')
		print(infoset_strategies_lvl0)
		print("lvl 1")
		print(infoset_strategies_lvl1)
		print("lvl 2")
		print(infoset_strategies_lvl2)

		print("infoset_acting_players")
		print("lvl 0")
		print(infoset_acting_player_lvl0)
		print("lvl 1")
		print(infoset_acting_player_lvl1)
		print("lvl 2")
		print(infoset_acting_player_lvl2)



		print("node_to_infoset")
		print("lvl 0")
		print(self.node_to_infoset[0])
		print("lvl 1")
		print(self.node_to_infoset[1])
		print("lvl 2")
		print(self.node_to_infoset[2])





if __name__ == '__main__':
	#input_line_chance = 'c "" 1 "" { "Ea (0.05)" 0.05 "Da (0.1)" 0.1 "Ca (0.1)" 0.1 "Ba (0.25)" 0.25 "Aa (0.5)" 0.5 } 1 "" { 0, 0 }'
	#input_line_player = 'p "" 2 4 "" { "Ja (0.9)" "Ia (0.1)" } 24 "" { 0, 0 }'
	#input_line_terminal = 't "" 38 "" { 10, -10 }'

	#gambit_efg_loader = GambitEFGLoader('doc/domain01_via_gambit.efg')
	#print("Chance:")
	#print(gambit_efg_loader.parse_node(input_line_chance))
	#print("Player:")
	#print(gambit_efg_loader.parse_node(input_line_player))
	#print("Terminal:")
	#print(gambit_efg_loader.parse_node(input_line_terminal))

	# exampel del
	#print('Del list')
	#a = [0, 1, 2, 3]
	#print(a)
	#del a[1]
	#print(a)

	#print('Del dict')
	#a = {'jedna': 1, 'dva': 2, 'tri': 3}
	#print(a)
	#del a['dva']
	#print(a)

	print("Muj print:")
	print(os.getcwd())

	gambit_efg_loader = GambitEFGLoader('/home/ruda/Documents/Projects/tensorcfr/TensorCFR/src/utils/domain01_via_gambit.efg')



