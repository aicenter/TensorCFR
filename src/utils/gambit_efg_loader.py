import re

NODE_TYPE_TERMINAL = 't'
NODE_TYPE_CHANCE = 'c'
NODE_TYPE_PLAYER = 'p'


class GambitEFGLoader:

	def __init__(self, input_efg_file):
		self.domain_filename = input_efg_file
		#self.domain_f = open(self.domain_filename)
		self.nodes = list()
		self.number_of_players = 2

	def parse_node(self, input_line):
		if len(input_line) == 0:
			return False

		node_type = input_line[0]

		if node_type == NODE_TYPE_CHANCE:
			return self.parse_chance_node(input_line)
		elif node_type == NODE_TYPE_PLAYER:
			return self.parse_player_node(input_line)
		elif node_type == NODE_TYPE_TERMINAL:
			return self.parse_terminal_node(input_line)
		else:
			return False

	def parse_actions(self, input_actions_str):
		parse_actions = re.findall(r'"(?P<name>[^"]*)" (?P<probability>[\d\.]+)', input_actions_str)
		return [{'name': action[0], 'probability': float(action[1])} for action in parse_actions]

	def parse_payoffs(self, input_payoffs_str):
		parse_payoffs = re.findall(r'\d+', input_payoffs_str)
		return [int(payoff) for payoff in parse_payoffs]

	def parse_chance_node(self, input_line):
		parse_line = re.search(
			r'^(?P<type>c) "(?P<name>[^"]*)" (?P<information_set_number>\d+) "(?P<information_set_name>[^"]*)" \{ (?P<actions_str>.*) \} (?P<outcome>\d+) \"\" \{ (?P<payoffs_str>.*) \}',
			input_line
		)

		actions = self.parse_actions(parse_line.group('actions_str'))
		payoffs = self.parse_payoffs(parse_line.group('payoffs_str'))

		return {
			'type': parse_line.group('type'),
			'name': parse_line.group('name'),
			'information_set_number': parse_line.group('information_set_number'),
			'information_set_name': parse_line.group('information_set_name'),
			'actions': actions,
			'outcome': parse_line.group('outcome'),
			'payoffs': payoffs
		}

	def parse_player_node(self, input_line):
		parse_line = re.search(
			r'^(?P<type>p) "(?P<name>[^"]*)" (?P<player_number>\d+) (?P<information_set_number>\d+) "(?P<information_set_name>[^"]*)" \{ (?P<actions_str>.*) \} (?P<outcome>\d+) \"\" \{ (?P<payoffs_str>.*) \}',
			input_line
		)

		actions = self.parse_actions(parse_line.group('actions_str'))
		payoffs = self.parse_payoffs(parse_line.group('payoffs_str'))

		return {
			'type': parse_line.group('type'),
			'name': parse_line.group('name'),
			'player_number': parse_line.group('player_number'),
			'information_set_number': parse_line.group('information_set_number'),
			'information_set_name': parse_line.group('information_set_name'),
			'actions': actions,
			'outcome': parse_line.group('outcome'),
			'payoffs': payoffs
		}

	def parse_terminal_node(self, input_line):
		parse_line = re.search(
			r'^(?P<type>t) "(?P<name>[^"]*)" (?P<outcome>\d+) \"\" \{ (?P<payoffs_str>.*) \}',
			input_line
		)

		payoffs = self.parse_payoffs(parse_line.group('payoffs_str'))

		return {
			'type': parse_line.group('type'),
			'name': parse_line.group('name'),
			'outcome': parse_line.group('outcome'),
			'payoffs': payoffs
		}


if __name__ == '__main__':
	input_line_chance = 'c "" 1 "" { "Ea (0.05)" 0.05 "Da (0.1)" 0.1 "Ca (0.1)" 0.1 "Ba (0.25)" 0.25 "Aa (0.5)" 0.5 } 1 "" { 0, 0 }'
	input_line_player = 'p "" 2 4 "" { "Ja (0.9)" "Ia (0.1)" } 24 "" { 0, 0 }'
	input_line_terminal = 't "" 38 "" { 10, -10 }'

	gambit_efg_loader = GambitEFGLoader('dummy')
	#print(gambit_efg_loader.parse_node(input_line_chance))
	#print(gambit_efg_loader.parse_node(input_line_player))
	print(gambit_efg_loader.parse_node(input_line_terminal))


"""
	search_result = re.search(
		r'^(?P<type>c) "(?P<name>[^"]*)" (?P<information_set_number>\d+) "(?P<information_set_name>[^"]*)" ',
		input_line)


	# parse chance witout payoffs and actions
	search_result = re.search(
		r'^(?P<type>c) "(?P<name>[^"]*)" (?P<information_set_number>\d+) "(?P<information_set_name>[^"]*)" \{ (?P<actions>.*) \} (?P<outcome>\d+) \"\" \{ (?P<payoffs>.*) \}',
		input_line)

	payoffs_str = search_result.group('payoffs')

	all_payoffs = re.findall(r'\d+', input_line)


input_line = "0, 0"
>>> search_result = re.findall(r' \d+,', input_line)inp
  File "<stdin>", line 1
    search_result = re.findall(r' \d+,', input_line)inp
                                                      ^
SyntaxError: invalid syntax
>>> search_result = re.findall(r' \d+,', input_line)
>>> search_result
[' 0,', ' 0,']
>>> search_result = re.findall(r' \d+', input_line)
>>> search_result
[' 0', ' 0']
>>> search_result = re.findall(r'\d+', input_line)
>>> search_result
['0', '0']
>>>

#get action name and probabilities
input_line = '"Ea (0.05)" 0.05 "Da (0.1)" 0.1 "Ca (0.1)" 0.1 "Ba (0.25)" 0.25 "Aa (0.5)" 0.5'

search_result = re.findall(r'"(?P<name>[^"]*)" (?P<probability>[\d\.]+)', input_line)

search_result = re.findall(r'"(?P<name>[^"]*)" (?P<probability>[\d\.]+)', input_line)
>>> search_result
[('Ea (0.05)', '0.05'), ('Da (0.1)', '0.1'), ('Ca (0.1)', '0.1'), ('Ba (0.25)', '0.25'), ('Aa (0.5)', '0.5')]
>>>



# priklad
del kus listu


"""