import numpy as np
import unittest
import tensorflow as tf

from src.utils.gambit_efg_loader import GambitEFGLoader


class TestGambibitEFGLoader(unittest.TestCase):
	def setUp(self):
		pass

	def tearDown(self):
		pass

	def test_parse_chance_node(self):
		input_str = 'c "" 1 "" { "Ea (0.05)" 0.05 "Da (0.1)" 0.1 "Ca (0.1)" 0.1 "Ba (0.25)" 0.25 "Aa (0.5)" 0.5 } 1 "" { 0, 0 }'

		gambit_loader = GambitEFGLoader("dummy")

		expected_output = {
			'payoffs': [0, 0],
			'outcome_name': '',
			'outcome': '1',
			'type': 'c',
			'name': '',
			'information_set_number': '1',
			'information_set_name': '',
			'actions': [
				{'probability': 0.05, 'name': 'Ea (0.05)'},
				{'probability': 0.1, 'name': 'Da (0.1)'},
				{'probability': 0.1, 'name': 'Ca (0.1)'},
				{'probability': 0.25, 'name': 'Ba (0.25)'},
				{'probability': 0.5, 'name': 'Aa (0.5)'}
			]
		}

		self.assertEqual(gambit_loader.parse_chance_node(input_str), expected_output)

if __name__ == '__main__':
	unittest.main()