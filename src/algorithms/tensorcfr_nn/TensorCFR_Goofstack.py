#!/usr/bin/env python3
import tensorflow as tf

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
#from src.algorithms.tensorcfr_nn.TensorCFR_NN import TensorCFR_NN
from src.commons.constants import DEFAULT_TOTAL_STEPS, DEFAULT_AVERAGING_DELAY
from src.domains.FlattenedDomain import FlattenedDomain
#from src.nn.NNMockUp import NNMockUp
from src.utils.cfr_utils import get_action_and_infoset_values
from src.utils.tf_utils import get_default_config_proto, print_tensor, masked_assign
from src.nn.data.postprocessing_ranges import load_nn
from src.nn.data.preprocessing_ranges import load_input_mask,load_output_mask,load_history_identifier,load_infoset_list,load_infoset_hist_ids,filter_by_card_combination,filter_by_public_state
import numpy as np

class TensorCFR_Goofstack(TensorCFRFixedTrunkStrategies):
	def __init__(self, domain: FlattenedDomain, neural_net=None, trunk_depth=0):
		##TODO investigate why it fails at trunk_depth = 0
		"""
		Constructor for an instance of TensorCFR_NN algorithm with given parameters (as a TensorFlow computation graph).

		:param domain: The domain of the game (as an instance of class `FlattenedDomain`). TensorCFR (the CFR+ algorithm
		with neural network for prediction of nodal expected values) will be launched for this game.
	  :param trunk_depth: The number of levels of the trunk where the strategies are kept fixed. It should be an integer between `0` to
		`self.domain.levels`. It defaults to `0` (no trunk).
		"""
		super().__init__(
			domain,
			trunk_depth,
			levels=trunk_depth + 1,  # `trunk_depth` levels where strategy is fixed + the final one at the bottom
			acting_depth=trunk_depth
		)

		self.neural_net =  load_nn(neural_net) if neural_net.__class__ == str else neural_net
		self.session = tf.Session(config=get_default_config_proto())
		self.construct_computation_graph()
		self.mask = load_input_mask()
		self.output_mask = load_output_mask()
		self.hist_id = load_history_identifier()
		self.infoset_list = load_infoset_list()
		self.infoset_hist_ids = load_infoset_hist_ids().iloc[:, :120]
		self.public_states_list = [(x, y, z) for x in [0, 1, -1] for y in [0, 1, -1] for z in [0, 1, -1]]
		self.tensor_cfr_in_mask = np.zeros(120 ** 2)
		with tf.variable_scope("initialization"):
			setup_messages, feed_dictionary = self.set_up_feed_dictionary(method="by-domain")
			print(setup_messages)
		self.average_strategies_over_steps = None
		self.session.run(tf.global_variables_initializer(), feed_dict=feed_dictionary)


	def update_strategy_of_updating_player(self, acting_player=None):  # override not to fix trunk
		"""
		Update for the strategy for the given `acting_player`.

		Args:
			:param acting_player: A variable. An index of the player whose strategies are to be updated.

		Returns:
			A corresponding TensorFlow operation (from the computation graph).
		"""
		if acting_player is None:
			acting_player = self.domain.current_updating_player
		infoset_strategies_matched_to_regrets = self.get_strategy_matched_to_regrets()
		infoset_acting_players = self.domain.get_infoset_acting_players()
		ops_update_infoset_strategies = [None] * self.acting_depth
		with tf.variable_scope("update_strategy_of_updating_player"):
			for level in range(self.acting_depth):
				with tf.variable_scope("level{}".format(level)):
					infosets_of_acting_player = tf.reshape(
						# `tf.reshape` to force "shape of 2D tensor" == [number of infosets, 1]
						tf.equal(infoset_acting_players[level], acting_player),
						shape=[self.domain.current_infoset_strategies[level].shape[0]],
						name="infosets_of_updating_player_lvl{}".format(level)
					)
					ops_update_infoset_strategies[level] = masked_assign(
						ref=self.domain.current_infoset_strategies[level],
						mask=infosets_of_acting_player,
						value=infoset_strategies_matched_to_regrets[level],
						name="op_update_infoset_strategies_lvl{}".format(level)
					)
			return ops_update_infoset_strategies

	def construct_computation_graph(self):
		self.cfr_step_op = self.do_cfr_step()
		self.input_ranges = self.get_trunk_nodal_ranges_p1_p2()

	def tensorcfr_to_nn_input(self,tensor_cfr_out=None):

		##TODO get ranges from tensorcfrfixestrunk. bring them into format [public_state,ranges p1] for each publicstate
		## TODO implement range of ifnoset in tensorcfr. its easier
		tensor_cfr_out = tensor_cfr_out.eval(session=self.session)
		mask = self.mask.copy()
		hist_id = self.hist_id.copy()

		for public_state in self.public_states_list:

			df_by_public_state = filter_by_public_state(hist_id, public_state)

			for cards in mask.columns[3:123]:
				## for player 1

				cards_df = filter_by_card_combination(df_by_public_state, cards, 1)

				if cards_df.shape[0] >= 1:

					# puts range of p1 in of infoset "cards" of public state "public_state" into mask

					mask.loc["".join(tuple(map(str, public_state))), cards] = float(
						tensor_cfr_out[cards_df.index[0], 0])

				else:

					mask.loc["".join(tuple(map(str, public_state))), cards] = 0

			for cards in mask.columns[123:]:

				cards_df = filter_by_card_combination(df_by_public_state, cards, 2)

				if cards_df.shape[0] == 1:

					mask.loc["".join(tuple(map(str, public_state))), cards] = float(
						tensor_cfr_out[cards_df.index[0], 1])



				elif cards_df.shape[0] > 1:

					mask.loc["".join(tuple(map(str, public_state))), cards] = float(
						tensor_cfr_out[cards_df.index[0], 1])


				elif cards_df.shape[0] == 0:

					mask.loc["".join(tuple(map(str, public_state))), cards] = 0

		return mask

	def nn_out_to_tensorcfr_in(self,nn_out=None):

		if nn_out.shape != (27, 120):
			raise ValueError

		else:
			## this version is only for nns that output cfv of p1. meaning a vector of size 120 for each public state

			tensor_cfr_in = self.tensor_cfr_in_mask.copy()

			for id in self.infoset_list:

				tensor_cfr_in[id] = nn_out[np.where(self.infoset_hist_ids == id)]

			return tensor_cfr_in

	def predict_lvl10_cf_values(self, input_ranges=None, name="predictions"):
		## TODO change to actual CFV. right now is u(x) not v(x)
		if input_ranges is None:
			input_ranges = self.input_ranges

		tensorcfr_in = self.tensorcfr_to_nn_input(input_ranges)

		nn_out = np.vstack([self.neural_net.predict(tensorcfr_in.values[i,:].reshape(1,243)) for i in range(tensorcfr_in.shape[0])])

		predicted_cf_values = self.nn_out_to_tensorcfr_in(nn_out)

		return tf.identity(predicted_cf_values, name=name)

	def get_nodal_cf_values(self, for_player=None):  # TODO insert nn preds at lvl 10 and should be fine
		"""
		Compute counterfactual values of nodes by (tensor-)multiplying reach probabilities and expected values.

		:param for_player: The player for which the counterfactual values are computed. These values are usually
		 computed for the updating player. Therefore, `for_player` is set to `current_updating_player` by default.
		:return: The counterfactual values of nodes based on `current_infoset_strategies`.
		"""
		expected_values = self.get_expected_values(for_player=for_player)
		reach_probabilities = self.get_nodal_reach_probabilities(for_player=for_player)
		with tf.variable_scope("nodal_counterfactual_values"):
			return [
				tf.multiply(
					reach_probabilities[level],
					expected_values[level],
					name="nodal_cf_value_lvl{}".format(level)
				) for level in range(self.levels)
			]

	def get_infoset_cf_values(self, for_player=None):  # TODO change this method to use predictions of network for lvl10
		"""
		Compute infoset(-action) counterfactual values by summing relevant counterfactual values of nodes.

		:param for_player: The player for which the counterfactual values are computed. These values are usually
		 computed for the updating player. Therefore, `for_player` is set to `current_updating_player` by default.
		:return: The infoset(-action) counterfactual values based on `current_infoset_strategies`.
		"""
		if for_player is None:
			player_name = "current_player"
		else:
			player_name = "player{}".format(for_player)
		nodal_cf_values = self.get_nodal_cf_values(for_player=for_player)
		infoset_actions_cf_values, infoset_cf_values = [], []
		with tf.variable_scope("infoset_actions_cf_values"):
			for level in range(self.acting_depth):
				with tf.variable_scope("level{}".format(level)):
					infoset_action_cf_value, infoset_cf_value = get_action_and_infoset_values(
						values_in_children=nodal_cf_values[level + 1],
						action_counts=self.action_counts[level],
						parental_node_to_infoset=self.domain.inner_node_to_infoset[level],
						infoset_strategy=self.domain.current_infoset_strategies[level],
						name="cf_values_lvl{}_for_{}".format(level, player_name)
					)
					infoset_cf_values.append(infoset_cf_value)
					infoset_actions_cf_values.append(infoset_action_cf_value)
		return infoset_actions_cf_values, infoset_cf_values


	def cf_values_lvl10_to_exp_values(self):

		nodal_reaches_lvl_10 = self.get_nodal_reach_probabilities(for_player=1)[10]
		cf_values_lvl_10 = self.predict_lvl10_cf_values()
		with tf.variable_scope("level{}".format(self.levels - 1)):
			cf_values_to_exp_values = tf.divide(cf_values_lvl_10,nodal_reaches_lvl_10,name="predictions_to_expected_values_lvl10")

		return cf_values_to_exp_values


	def construct_lowest_expected_values(self, player_name, signum):
		with tf.variable_scope("level{}".format(self.levels - 1)):

			self.expected_values[self.levels - 1] = tf.multiply(
				signum,
				self.cf_values_lvl10_to_exp_values(),
				name="expected_values_lvl{}_for_{}".format(self.levels - 1, player_name)
			)


	def cf_nn_preds_to_cf_values(self):
		##TODO put cf preds of nn in list of cf values
		pass



	def run_cfr(self, total_steps=DEFAULT_TOTAL_STEPS, delay=DEFAULT_AVERAGING_DELAY, verbose=False,
	            register_strategies_on_step=None):
		if register_strategies_on_step is None:
			register_strategies_on_step = [total_steps - 1]  # by default, register just the last iteration
		self.average_strategies_over_steps = dict()        # reset the dict

		self.cfr_parameters = {
			"total_steps"    : total_steps,
			"averaging_delay": delay,
			"trunk_depth"    : self.trunk_depth,
		}
		self.set_up_cfr_parameters(delay, total_steps)
		self.set_log_directory()
		with tf.summary.FileWriter(self.log_directory, tf.get_default_graph()):
			for step in range(total_steps):
				print("\n########## CFR step {} ##########".format(step))
				predicted_cf_values = self.predict_lvl10_cf_values()
				if verbose:
					print("Before:")
					print_tensor(self.session, self.input_ranges)
					print_tensor(self.session, predicted_cf_values)
				np_predicted_equilibrial_values = self.session.run(predicted_cf_values)
				self.session.run(self.cfr_step_op, {self.predicted_cf_values: np_predicted_equilibrial_values})
				if verbose:
					print("After:")
					print_tensor(self.session, self.input_ranges)

				if step in register_strategies_on_step:
					self.average_strategies_over_steps["average_strategy_step{}".format(step)] = [
						self.session.run(strategy).tolist() for strategy in self.average_infoset_strategies[:self.trunk_depth]
					]
