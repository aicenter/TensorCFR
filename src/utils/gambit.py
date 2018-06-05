import re

from src.commons import constants


class NotAcceptableFormatException(Exception):
	pass


class NotRecognizedPlayersException(Exception):
	pass


class NotRecognizedTreeNodeException(Exception):
	pass


class NotImplementedFormatException(Exception):
	pass


class Parser:
	def __init__(self):
		pass

	@staticmethod
	def parse_header(input_line):
		results = re.search(r'^(?P<format>EFG|NFG) (?P<version>\d) R "(?P<name>[^"]+)" \{(?P<players_dirty>.*)\}',
							input_line)
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

	def parse_player_node(self, input_line):
		parse_line = re.search(
			r'^(?P<type>' + constants.GAMBIT_NODE_TYPE_PLAYER + ') "(?P<name>[^"]*)" (?P<player_number>\d+) (?P<information_set_number>\d+)\ ?"?(?P<information_set_name>[^"]*)"?\ ?\{?(?P<actions_optional>[^\}]*)\}?\ ?(?P<outcome>\d+)\ ?"?(?P<outcome_name>[^"]*)"?\ ?\{?(?P<payoffs_optional>[^\}]*)\}?',
			input_line
		)

		actions = self.parse_actions_player(parse_line.group('actions_optional'))
		payoffs = self.parse_payoffs(parse_line.group('payoffs_optional'))
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
			r'^(?P<type>' + constants.GAMBIT_NODE_TYPE_TERMINAL + ') "(?P<name>[^"]*)" (?P<outcome>\d+)\ ?"?(?P<outcome_name_optional>[^"]*)"?\ ?\{(?P<payoffs>.*)\}',
			input_line
		)

		payoffs = self.parse_payoffs(parse_line.group('payoffs'))
		# infoset_id = 't-' + str(self.terminal_nodes_cnt)
		infoset_id = 't'

		self.terminal_nodes_cnt += 1

		return {
			'type': parse_line.group('type'),
			'name': parse_line.group('name'),
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