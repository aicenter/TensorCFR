import re

from src.commons import constants

from .exceptions import NotAcceptableFormatException, NotRecognizedPlayersException, NotRecognizedTreeNodeException, NotImplementedFormatException


class Parser:

	@staticmethod
	def parse_header(input_line):
		results = re.search(
			r'^(?P<format>EFG|NFG) (?P<version>\d) R "(?P<name>[^"]+)" {(?P<players_dirty>.*)}',
			input_line
		)
		if results:
			results_players = re.findall(r'"([^"]+)"', results.group('players_dirty'))
			if results_players is None:
				raise NotRecognizedPlayersException
			return_dict = {
				'format': results.group('format'),
				'version': int(results.group('version')),
				'name': results.group('name'),
				'players': results_players
			}
			return return_dict
		else:
			raise NotAcceptableFormatException

	@staticmethod
	def parse_node(input_line):
		# http://www.gambit-project.org/gambit13/formats.html
		if len(input_line) == 0:
			return False

		node_type = input_line[0]

		if node_type == constants.GAMBIT_NODE_TYPE_CHANCE:
			return Parser.parse_chance_node(input_line)
		elif node_type == constants.GAMBIT_NODE_TYPE_PLAYER:
			return Parser.parse_player_node(input_line)
		elif node_type == constants.GAMBIT_NODE_TYPE_TERMINAL:
			return Parser.parse_terminal_node(input_line)
		else:
			return False

	@staticmethod
	def parse_probability(probability_str):
		if '/' in probability_str:
			probability_list = probability_str.split('/')
			return float(int(probability_list[0]) / int(probability_list[1]))
		else:
			return float(probability_str)

	def parse_actions_chance(input_actions_str):
		parse_actions = re.findall(r'"(?P<name>[^"]*)" (?P<probability>[\d\./]+)', input_actions_str)
		return [{'name': action[0], 'probability': Parser.parse_probability(action[1])} for action in parse_actions]

	@staticmethod
	def parse_actions_player(input_actions_str):
		parse_actions = re.findall(r'"(?P<name>[^"]*)"', input_actions_str)
		return [{'name': action[0]} for action in parse_actions]

	@staticmethod
	def parse_payoffs(input_payoffs_str):
		parse_payoffs = re.findall(r'[\-]?[\d]+', input_payoffs_str)
		return [int(payoff) for payoff in parse_payoffs]

	@staticmethod
	def parse_chance_node(input_line):
		parse_line = re.search(
			r'^(?P<type>' + constants.GAMBIT_NODE_TYPE_CHANCE + ') "(?P<name>[^"]*)" (?P<information_set_number>\d+)\ ?"?(?P<information_set_name_optional>[^"]*)"?\ ?\{?(?P<actions_optional>[^\}]*)\}?\ ?(?P<outcome>\d+)\ ?"?(?P<outcome_name_optional>[^"]*)"?\ ?\{?(?P<payoffs_optional>.*)\}?',
			input_line
		)

		actions = Parser.parse_actions_chance(parse_line.group('actions_optional'))
		payoffs = Parser.parse_payoffs(parse_line.group('payoffs_optional'))
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
			'tensorcfr_id': constants.CHANCE_PLAYER,
			'infoset_id': infoset_id
		}

	@staticmethod
	def parse_player_node(input_line):
		parse_line = re.search(
			r'^(?P<type>' + constants.GAMBIT_NODE_TYPE_PLAYER + ') "(?P<name>[^"]*)" (?P<player_number>\d+) (?P<information_set_number>\d+)\ ?"?(?P<information_set_name>[^"]*)"?\ ?\{?(?P<actions_optional>[^\}]*)\}?\ ?(?P<outcome>\d+)\ ?"?(?P<outcome_name>[^"]*)"?\ ?\{?(?P<payoffs_optional>[^\}]*)\}?',
			input_line
		)

		actions = Parser.parse_actions_player(parse_line.group('actions_optional'))
		payoffs = Parser.parse_payoffs(parse_line.group('payoffs_optional'))
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
			'tensorcfr_id': int(parse_line.group('player_number')), # TODO smazat, v terminalu se pouziva misto player number
			'infoset_id': infoset_id
		}

	@staticmethod
	def parse_terminal_node(input_line):
		parse_line = re.search(
			r'^(?P<type>' + constants.GAMBIT_NODE_TYPE_TERMINAL + ') "(?P<name>[^"]*)" (?P<outcome>\d+)\ ?"?(?P<outcome_name_optional>[^"]*)"?\ ?\{(?P<payoffs>.*)\}',
			input_line
		)

		payoffs = Parser.parse_payoffs(parse_line.group('payoffs'))
		# infoset_id = 't-' + str(self.terminal_nodes_cnt)
		infoset_id = 't'

		return {
			'type': parse_line.group('type'),
			'name': parse_line.group('name'),
			'actions': [],
			'outcome': parse_line.group('outcome'),
			'outcome_name': parse_line.group('outcome_name_optional'),
			'payoffs': payoffs,
			'tensorcfr_id': constants.NO_ACTING_PLAYER,
			'infoset_id': infoset_id
		}

	@staticmethod
	def is_gambit_node(line):
		if line[0] == constants.GAMBIT_NODE_TYPE_CHANCE or line[0] == constants.GAMBIT_NODE_TYPE_PLAYER or line[
			0] == constants.GAMBIT_NODE_TYPE_TERMINAL:
			return True
		return False


class Parser2:
	def __init__(self, file):
		print("__init__")
		self.__gambit_file = open(file)

		print("Parser2 hello world")
		line = self.__gambit_file.readline()

		if line.startswith('<'):
			flag_is_efg_file = False
			for line in self.__gambit_file:
				if line.strip() == "<efgfile>":
					flag_is_efg_file = True
					break
			if not flag_is_efg_file:
				raise NotImplementedFormatException()
		elif len(line) > 3 and line[0:3] == "EFG":
			pass
		elif len(line) > 3 and 	line[0:3] == "NFG":
			raise NotImplementedFormatException()
		else:
			raise NotImplementedError()

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.__gambit_file.close()
		return False

	def __parse_header(self, header):
		print("parse_header")
		print(header)

	def next_node(self):
		for line in self.__gambit_file:
			if line.strip() == "</efgfile>":
				break
			elif not (line.startswith('t') or line.startswith('p') or line.startswith('c')):
				continue
			else:
				yield Parser.parse_node(line.strip())


if __name__ == "__main__":
	# p = Parser2("/home/ruda/Desktop/pokus_gambit.gbt")
	# print("next_node")
	#
	# for line in p.next_node():
	# 	print(line)

	gbt_desktop = "/home/ruda/Desktop/pokus_gambit.gbt"
	efg_domain01 = "/home/ruda/Documents/Projects/tensorcfr/TensorCFR/doc/domain01_via_gambit.efg"

	with Parser2(efg_domain01) as p:
		for line in p.next_node():
			print("in for line ")
			print(line)
