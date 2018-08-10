import re

from src.commons import constants as common_constants
from ..gambit_flattened_domains import constants
from ..gambit_flattened_domains import exceptions


class GambitNode:
	def __init__(self, node):
		self.type = self.__set_attr('type', node)
		self.name = self.__set_attr('name', node)
		self.information_set_number = self.__set_attr('information_set_number', node)
		self.information_set_name = self.__set_attr('information_set_name', node)
		self.actions = self.__set_attr('actions', node)
		self.outcome = self.__set_attr('outcome', node)
		self.outcome_name = self.__set_attr('outcome_name', node)
		self.payoffs = self.__set_attr('payoffs', node)
		self.tensorcfr_id = self.__set_attr('tensorcfr_id', node)
		self.information_set_id = self.__set_attr('infoset_id', node)
		self.player_number = self.__set_attr('player_number', node)

	def __str__(self):
		ret = ""
		if self.is_chance():
			ret = "Chance <{}, {}>".format(self.information_set_number, self.tensorcfr_id)
		if self.is_player():
			ret = "Player <{}, {}>".format(self.information_set_number, self.tensorcfr_id)
		if self.is_terminal():
			ret = "Terminal"
		return ret

	def __set_attr(self, key, dictionary):
		return dictionary[key] if key in dictionary else None

	def is_chance(self):
		return self.type == constants.CHANCE_NODE

	def is_player(self):
		return self.type == constants.PLAYER_NODE

	def is_terminal(self):
		return self.type == constants.TERMINAL_NODE


class Parser:
	def __init__(self, file):
		self.__gambit_file = open(file)

		self.header = self.__parse_header()

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.__gambit_file.close()
		return False

	def __parse_header(self):
		line = self.__gambit_file.readline()

		if line.startswith('<'):
			flag_is_efg_file = False
			for line in self.__gambit_file:
				if line.strip() == "<efgfile>":
					flag_is_efg_file = True
					return self.__parse_header_line(line)
			if not flag_is_efg_file:
				raise exceptions.NotImplementedFormatException()
		elif len(line) > 3 and line[0:3] == "EFG":
			return self.__parse_header_line(line)
		elif len(line) > 3 and line[0:3] == "NFG":
			raise exceptions.NotImplementedFormatException()
		else:
			raise NotImplementedError()

	def __parse_header_line(self, input_line):
		results = re.search(
			r'^(?P<format>EFG|NFG) (?P<version>\d) R "(?P<name>[^"]+)" ({(?P<players_dirty>[^}]*)}) ?({(?P<domain_parameters_dirty>.*)})?',
			input_line.strip()
		)
		if results:
			results_players = re.findall(r'"([^"]+)"', results.group('players_dirty'))
			if results.group('domain_parameters_dirty') is not None:
				result_domain_parameters = re.findall(r'"([^"]+)"', results.group('domain_parameters_dirty'))
			else:
				result_domain_parameters = list()
			if results_players is None:
				raise exceptions.NotRecognizedPlayersException()
			return_dict = {
				'format': results.group('format'),
				'version': int(results.group('version')),
				'name': results.group('name'),
				'players': results_players,
				'domain_parameters': result_domain_parameters
			}
			return return_dict
		else:
			raise exceptions.NotAcceptableFormatException()

	def __parse_node(self, input_line):
		# http://www.gambit-project.org/gambit13/formats.html
		if len(input_line) == 0:
			return False

		node_type = input_line[0]

		if node_type == constants.CHANCE_NODE:
			return self.__parse_chance_node(input_line)
		elif node_type == constants.PLAYER_NODE:
			return self.__parse_player_node(input_line)
		elif node_type == constants.TERMINAL_NODE:
			return self.__parse_terminal_node(input_line)
		else:
			return False

	def __parse_probability(self, probability_str):
		if '/' in probability_str:
			probability_list = probability_str.split('/')
			return float(int(probability_list[0]) / int(probability_list[1]))
		else:
			return float(probability_str)

	def __parse_actions_chance(self, input_actions_str):
		parse_actions = re.findall(r'"(?P<name>[^"]*)" (?P<probability>[\d\./]+)', input_actions_str)
		return [{'name': action[0], 'probability': self.__parse_probability(action[1])} for action in parse_actions]

	def __parse_actions_player(self, input_actions_str):
		parse_actions = re.findall(r'"(?P<name>[^"]*)"', input_actions_str)
		return [{'name': action[0]} for action in parse_actions]

	def __parse_payoffs(self, input_payoffs_str):
		parse_payoffs = re.findall(r'[\-]?[\d]+', input_payoffs_str)
		return [int(payoff) for payoff in parse_payoffs]

	def __parse_chance_node(self, input_line):
		parse_line = re.search(
			r'^(?P<type>' + constants.CHANCE_NODE + ') "(?P<name>[^"]*)" (?P<information_set_number>\d+)\ ?"?(?P<information_set_name_optional>[^"]*)"?\ ?\{?(?P<actions_optional>[^\}]*)\}?\ ?(?P<outcome>\d+)\ ?"?(?P<outcome_name_optional>[^"]*)"?\ ?\{?(?P<payoffs_optional>.*)\}?',
			input_line
		)

		actions = self.__parse_actions_chance(parse_line.group('actions_optional'))
		payoffs = self.__parse_payoffs(parse_line.group('payoffs_optional'))
		infoset_id = 'c-' + parse_line.group('information_set_number')

		return {
			'type': parse_line.group('type'),
			'name': parse_line.group('name'),
			'information_set_number': int(parse_line.group('information_set_number')),
			'information_set_name': parse_line.group('information_set_name_optional'),
			'actions': actions,
			'outcome': parse_line.group('outcome'),
			'outcome_name': parse_line.group('outcome_name_optional'),
			'payoffs': payoffs,
			'tensorcfr_id': common_constants.CHANCE_PLAYER,
			'infoset_id': infoset_id
		}

	def __parse_player_node(self, input_line):
		parse_line = re.search(
			r'^(?P<type>' + constants.PLAYER_NODE + ') "(?P<name>[^"]*)" (?P<player_number>\d+) (?P<information_set_number>\d+)\ ?"?(?P<information_set_name>[^"]*)"?\ ?\{?(?P<actions_optional>[^\}]*)\}?\ ?(?P<outcome>\d+)\ ?"?(?P<outcome_name>[^"]*)"?\ ?\{?(?P<payoffs_optional>[^\}]*)\}?',
			input_line
		)

		actions = self.__parse_actions_player(parse_line.group('actions_optional'))
		payoffs = self.__parse_payoffs(parse_line.group('payoffs_optional'))
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

	def __parse_terminal_node(self, input_line):
		parse_line = re.search(
			r'^(?P<type>' + constants.TERMINAL_NODE + ') "(?P<name>[^"]*)" (?P<outcome>\d+)\ ?"?(?P<outcome_name_optional>[^"]*)"?\ ?\{(?P<payoffs>.*)\}',
			input_line
		)

		payoffs = self.__parse_payoffs(parse_line.group('payoffs'))
		# infoset_id = 't-' + str(self.terminal_nodes_cnt)
		infoset_id = 't'

		return {
			'type': parse_line.group('type'),
			'name': parse_line.group('name'),
			'actions': [],
			'outcome': parse_line.group('outcome'),
			'outcome_name': parse_line.group('outcome_name_optional'),
			'payoffs': payoffs,
			'tensorcfr_id': common_constants.NO_ACTING_PLAYER,
			'infoset_id': infoset_id
		}

	def next_node(self):
		for line in self.__gambit_file:
			if line.strip() == "</efgfile>":
				break
			elif not (line.startswith('t') or line.startswith('p') or line.startswith('c')):
				continue
			else:
				yield GambitNode(self.__parse_node(line.strip()))


if __name__ == "__main__":
	import os

	domain01_gambit_efg = os.path.join(
		common_constants.PROJECT_ROOT,
		"doc",
		"domain01_via_gambit.efg"
	)

	iigs_efg = os.path.join(
		common_constants.PROJECT_ROOT,
		"doc",
		"goofspiel",
		"II-GS3.efg"
	)

	cnt = 1
	with Parser(iigs_efg) as p:
		print(p.header)
		for node in p.next_node():
			print("-----------------------------")
			print(cnt)
			print(node)
			print("type: " + str(node.type)) # pouziva se vsude
			print("name: " + str(node.name)) # nepouziva se v gambit_loader, v gambit.py se pouziva
			print("information_set_number: " + str(node.information_set_number)) # nepouziva se v gambit_loader
			print("information_set_name: " + str(node.information_set_name)) # nepouziva se nikde
			print("actions: " + str(node.actions)) # pouziva se vsude
			print("outcome: " + str(node.outcome)) #nepouziva se
			print("outcome_name: " + str(node.outcome_name)) #nepouziva se
			print("payoffs: " + str(node.payoffs)) # pouziva se
			print("tensorcfr_id: " + str(node.tensorcfr_id)) # pouziva se v gambit_loader (?)
			print("information_set_id: " + str(node.information_set_id)) # pouziva se
			print("player_number: " + str(node.player_number)) # nepouziva se v gambit.py,
			cnt += 1