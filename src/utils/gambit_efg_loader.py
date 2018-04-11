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
			raise NotImplemented
		elif node_type == NODE_TYPE_TERMINAL:
			raise NotImplemented
		else:
			return False

	def parse_chance_node(self, input_line):
		parse_line = re.search(
			r'^(?P<type>c) "(?P<name>[^"]*)" (?P<information_set_number>\d+) "(?P<information_set_name>[^"]*)" \{ (?P<actions_str>.*) \} (?P<outcome>\d+) \"\" \{ (?P<payoffs_str>.*) \}',
			input_line
		)
		parse_actions = re.findall(r'"(?P<name>[^"]*)" (?P<probability>[\d\.]+)', parse_line.group('actions_str'))
		parse_payoffs = re.findall(r'\d+', parse_line.group('payoffs_str'))

		actions = [{'name': action[0], 'probability': float(action[1])} for action in parse_actions]
		payoffs = [int(payoff) for payoff in parse_payoffs]

		return {
			'type': parse_line.group('type'),
			'name': parse_line.group('name'),
			'information_set_number': parse_line.group('information_set_number'),
			'information_set_name': parse_line.group('information_set_name'),
			'actions': actions,
			'outcome': parse_line.group('outcome'),
			'payoffs': payoffs
		}


if __name__ == '__main__':
	input_line = 'c "" 1 "" { "Ea (0.05)" 0.05 "Da (0.1)" 0.1 "Ca (0.1)" 0.1 "Ba (0.25)" 0.25 "Aa (0.5)" 0.5 } 1 "" { 0, 0 }'

	gambit_efg_loader = GambitEFGLoader('pokus')
	print(gambit_efg_loader.parse_node(input_line))


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