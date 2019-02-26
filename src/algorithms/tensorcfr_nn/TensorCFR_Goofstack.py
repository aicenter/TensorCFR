#!/usr/bin/env python3
import tensorflow as tf

from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import DEFAULT_TOTAL_STEPS, DEFAULT_AVERAGING_DELAY, FLOAT_DTYPE, INT_DTYPE, PLAYER1, PLAYER2
from src.domains.FlattenedDomain import FlattenedDomain
from src.utils.tf_utils import get_default_config_proto, print_tensor, masked_assign
from src.nn.data.postprocessing_ranges import load_nn
from src.nn.data.preprocessing_ranges import load_input_mask,load_output_mask,load_history_identifier,filter_by_card_combination,\
	filter_by_public_state,load_auginfoset_dict,load_infoset_dict
import numpy as np
from copy import deepcopy
from random import choice

class TensorCFR_Goofstack(TensorCFRFixedTrunkStrategies):
	def __init__(self, domain: FlattenedDomain, neural_net=None, trunk_depth=10):

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
		self.public_states_list = [(x, y, z) for x in [0, 1, -1] for y in [0, 1, -1] for z in [0, 1, -1]]
		self.tensor_cfr_in_mask = np.zeros(120 ** 2)
		self.infset_dict = load_infoset_dict()
		self.auginfset_dict = load_auginfoset_dict()
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


	def non_zero_reach_node_of_auginfset(self):
		reaches = self.get_nodal_reach_probabilities(for_player=self.domain.current_updating_player)[self.trunk_depth]
		#reaches = self.get_nodal_reaches_at_trunk_depth()
		#reaches_flat = self.session.run(tf.reshape(reaches, [-1]))
		#print("sumofreaches: {}".format(sum(reaches_flat)))
		bool_non_zero_reaches = tf.reshape(tf.where(tf.not_equal(reaches,0,name="bool_non_zero_reaches_lvl10")),[-1])
		#a = self.session.run(bool_non_zero_reaches)
		#print("{} non zero reach trunk histories".format(a.shape[0]))
		#print_tensor(self.session,bool_non_zero_reaches)

		#dict_operation = tf.cond(lambda : self.check_player(),lambda: self.inf_set_cond(bool_non_zero_reaches),
		# lambda: self.auginf_set_cond(bool_non_zero_reaches))
		cond = self.session.run(self.check_player())
		print("player {}".format(cond))
		if cond:

			print("infoset")
			infsetdict = deepcopy(self.infset_dict)
			np_bool_non_zero_reaches = bool_non_zero_reaches.eval(session=self.session)
			for key, value in infsetdict.items():
				infsetdict[key] = [idx for idx in value if idx in np_bool_non_zero_reaches]

			return infsetdict

		else:

			print("auginfoset")
			auginfsetdict = deepcopy(self.auginfset_dict)
			np_bool_non_zero_reaches = bool_non_zero_reaches.eval(session=self.session)
			for key, value in auginfsetdict.items():
				auginfsetdict[key] = [idx for idx in value if idx in np_bool_non_zero_reaches]

			return auginfsetdict

	def check_player(self):

		return tf.equal(self.domain.current_updating_player, tf.constant(value=PLAYER1))

	def tensorcfr_to_nn_input(self,tensor_cfr_out=None):
		##TODO check if taking index 0 of each seed is valid. maybe its sum or take one that is not 0? ask vojta

		##TODO use tf.scatter_nd for this. Could be much faster. just get all the first indices for each infset.
		## TODO implement range of ifnoset in tensorcfr. its easier
		tensor_cfr_out = tensor_cfr_out.eval(session=self.session)
		mask = deepcopy(self.mask)
		hist_id = deepcopy(self.hist_id)

		for public_state in deepcopy(self.public_states_list):

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
		##TODO use ordered dict to be able to hav i-th key of dict correspond to i-th index of output vector of nn
		##TODO again use tf.scatter_nd to put values by indices
		## this version is only for nns that output cfv of p1. meaning a vector of size 120 for each public state

		idxdict = self.non_zero_reach_node_of_auginfset()

		print(" {} non zero reah h in dict".format(sum([value.__len__() for key,value in idxdict.items()])))

		tensor_cfr_in = deepcopy(self.tensor_cfr_in_mask)

		for loc,idx in idxdict.items():

			if idx.__len__() != 0:

				myloctuple = tuple(map(int, loc[1:-1].split(',')))

				tensor_cfr_in[choice(idx)] = nn_out[myloctuple[0],myloctuple[1]]

			else:
				continue

		return tensor_cfr_in

	def predict_lvl10_cf_values(self, input_ranges=None, name="predictions"):
		## TODO change to actual CFV. right now is u(x) not v(x)
		## TODO output dim of network will be 240
		## TODO create map of histories in augmented infosets lvl 10
		## TODO choose history with non zero reach (action prob)
		## TODO create if check which player is currently updating / cumulating
		## TODO if current updating player = 1 do padding of 2 and vice versa
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
		expected_values = self.get_expected_values()
		reach_probabilities = self.get_nodal_reach_probabilities(for_player=for_player)
		#lowest_cf_values = self.construct_lowest_cf_values()
		with tf.variable_scope("nodal_counterfactual_values"):
			return [
				tf.multiply(
					reach_probabilities[level],
					expected_values[level],
					name="nodal_cf_value_lvl{}".format(level)
				) for level in range(self.levels)]#+[lowest_cf_values]
	#
	# def get_infoset_cf_values(self, for_player=None):  # TODO change this method to use predictions of network for lvl10
	# 	"""
	# 	Compute infoset(-action) counterfactual values by summing relevant counterfactual values of nodes.
	#
	# 	:param for_player: The player for which the counterfactual values are computed. These values are usually
	# 	 computed for the updating player. Therefore, `for_player` is set to `current_updating_player` by default.
	# 	:return: The infoset(-action) counterfactual values based on `current_infoset_strategies`.
	# 	"""
	# 	if for_player is None:
	# 		player_name = "current_player"
	# 	else:
	# 		player_name = "player{}".format(for_player)
	# 	nodal_cf_values = self.get_nodal_cf_values(for_player=for_player)
	# 	infoset_actions_cf_values, infoset_cf_values = [], []
	# 	with tf.variable_scope("infoset_actions_cf_values"):
	# 		for level in range(self.acting_depth):
	# 			with tf.variable_scope("level{}".format(level)):
	# 				infoset_action_cf_value, infoset_cf_value = get_action_and_infoset_values(
	# 					values_in_children=nodal_cf_values[level + 1],
	# 					action_counts=self.action_counts[level],
	# 					parental_node_to_infoset=self.domain.inner_node_to_infoset[level],
	# 					infoset_strategy=self.domain.current_infoset_strategies[level],
	# 					name="cf_values_lvl{}_for_{}".format(level, player_name)
	# 				)
	# 				infoset_cf_values.append(infoset_cf_value)
	# 				infoset_actions_cf_values.append(infoset_action_cf_value)
	# 	return infoset_actions_cf_values, infoset_cf_values
	#
	def cf_values_to_exp_values(self,cfreaches,cfvalues):
		# cf_values_to_exp_values = tf.where(condition=tf.equal(nodal_reaches_lvl_10_float64,0.0),
		# 	                                   x=tf.zeros_like(cf_values_lvl_10,dtype=tf.float64),
		# 	                                   y=tf.divide(cf_values_lvl_10,nodal_reaches_lvl_10_float64),
		# 	                                                  name="predictions_to_expected_values_lvl10")
		##TODO remember i put cfvalues to exp util right now if the nodal reach is 0
		nodal_reaches_lvl_10_float64 = tf.cast(cfreaches, tf.float64)
		return tf.where(condition=tf.equal(nodal_reaches_lvl_10_float64,0.0),x=tf.zeros_like(cfvalues,dtype=tf.float64),y=tf.divide(cfvalues,nodal_reaches_lvl_10_float64))

	def cf_values_lvl10_to_exp_values(self):

		nodal_reaches_lvl_10 = self.get_nodal_reach_probabilities(for_player=self.domain.current_updating_player)[self.trunk_depth]

		cf_values_lvl_10 = self.predict_lvl10_cf_values()

		return tf.identity(self.cf_values_to_exp_values(nodal_reaches_lvl_10,cf_values_lvl_10),name="cf_values_to_exp_values")

	def get_expected_values(self):
		"""
		Compute expected values of nodes using the bottom-up tree traversal.

		:param for_player: The player for which the expected values are computed. These values are usually computed for the
		 updating player when counterfactual values are computed. Therefore, by default the expected values are computed for
		 the `current_updating_player`, i.e. multiplied with `signum` of `signum_of_current_player`.
		:return: The expected values of nodes based on `current_infoset_strategies`.
		"""

		#player_name = "player{}".format(self.domain.current_updating_player)
		player_name = "current_player"
		node_strategies = self.get_node_strategies()
		with tf.variable_scope("expected_values"):
			self.construct_lowest_expected_values(player_name)
			for level in reversed(range(self.levels - 1)):
				with tf.variable_scope("level{}".format(level)):
					weighted_sum_of_values = tf.segment_sum(
						##TODO sum of exp values times all action probs of parent that reach to all h in I
						data=node_strategies[level + 1] * self.expected_values[level + 1],
						segment_ids=self.domain.parents[level + 1],
						name="weighted_sum_of_values_lvl{}".format(level),
					)
					scatter_copy_indices = tf.expand_dims(
						tf.cumsum(
							tf.ones_like(weighted_sum_of_values, dtype=INT_DTYPE),
							exclusive=True,
						),
						axis=-1,
						name="scatter_copy_indices_lvl{}".format(level)
					)
					extended_weighted_sum = tf.scatter_nd(
						indices=scatter_copy_indices,
						updates=weighted_sum_of_values,
						shape=self.domain.utilities[level].shape,
						name="extended_weighted_sum_lvl{}".format(level)
					)
					self.expected_values[level] = tf.where(
						condition=self.domain.mask_of_inner_nodes[level],
						x=extended_weighted_sum,
						y=tf.reshape(
							self.domain.utilities[level],
							shape=[self.domain.utilities[level].shape[-1]],
						),
						name="expected_values_lvl{}_for_{}".format(level, player_name)
					)
		return self.expected_values

	def construct_lowest_expected_values(self, player_name):
		with tf.variable_scope("level{}".format(self.levels - 1)):
			lowest_utilities = self.domain.utilities[self.levels - 1]
			self.predicted_to_exp_values = tf.placeholder_with_default(
				lowest_utilities,
				shape=lowest_utilities.shape,
				name="predicted_to_exp_values"
			)
			# self.predicted_to_exp_values = tf.Variable(
			# 	lowest_utilities,
			#
			# 	name="predicted_to_exp_values"
			# )
	## changed signum to 1 since new network will predict for each player

			self.expected_values[self.levels - 1] = tf.multiply(1.0,self.predicted_to_exp_values,
				name="expected_values_lvl{}_for_{}".format(self.levels - 1, player_name)
			)

	def construct_lowest_cf_values(self):
		with tf.variable_scope("level{}".format(self.levels - 1)):
			lowest_utilities = self.domain.utilities[self.levels - 1]
			self.predicted_cf_values = tf.placeholder_with_default(
				lowest_utilities,
				shape=lowest_utilities.shape,
				name="predicted_cf_values"
			)
			# self.predicted_to_exp_values = tf.Variable(
			# 	lowest_utilities,
			#
			# 	name="predicted_to_exp_values"
			# )
	## changed signum to 1 since new network will predict for each player

			return tf.multiply(1.0,self.predicted_cf_values)


	def run_cfr(self, total_steps=DEFAULT_TOTAL_STEPS, delay=DEFAULT_AVERAGING_DELAY, verbose=True,
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
		#ops_list = []
		with tf.summary.FileWriter(self.log_directory, tf.get_default_graph()):
			for step in range(total_steps):
				print("\n########## CFR step {} ##########".format(step))
				print("current updating player is:")
				print_tensor(self.session,self.domain.current_updating_player)
				#predicted_to_exp_values =
				if verbose:
					print("# graph ops:{}".format(len(self.session.graph.get_operations())))

					#ops_list.append([self.session.graph.get_operations()])

					# if step > 0:
					# 	print([op for op in ops_list[step] if op not in ops_list[step-1]])

					#print("player owning utilities")
					#print_tensor(self.session,self.domain.player_owning_the_utilities)
					##TODO multiply cfv by range and sum them up
					print("Before:")
					a = self.session.run(self.get_trunk_nodal_ranges_p1_p2())
					print("non zero ranges p1: {}".format(np.count_nonzero(a[:,0])))
					print("non zero ranges p2: {}".format(np.count_nonzero(a[:,1])))
					#print("input ranges shape:{}.".format(a.shape))
					#print_tensor(self.session, self.input_ranges)
					#print_tensor(self.session, predicted_to_exp_values)

					#cfv_pred = self.session.run(self.predict_lvl10_cf_values())
					#bool_non_zero_cfv = tf.reshape(tf.where(tf.not_equal(cfv_pred, 0, name="bool_non_zero_cfv_lvl10")),[-1])
					#c = self.session.run(bool_non_zero_cfv)
					#print("{} non zero cfv trunk histories".format(c.shape[0]))
					#bool_non_zero_utils = tf.reshape(tf.where(tf.not_equal(predicted_to_exp_values, 0, name="bool_non_zero_util_lvl10")),[-1])
					#b = self.session.run(bool_non_zero_utils)
					#print("{} non zero exp util trunk histories".format(b.shape[0]))
					#print("current updating player:")
					#print_tensor(self.session, self.domain.current_updating_player)
					#print("with sign:")
					#print_tensor(self.session, self.domain.signum_of_current_player)
				np_predicted_cf_values = self.session.run(self.predict_lvl10_cf_values())
				#np_predicted_equilibrial_values = self.session.run(self.cf_values_lvl10_to_exp_values())

				#cf_preds = self.session.run(self.predict_lvl10_cf_values())
				#print("sum of exp values for both players: {}".format(np.count_nonzero(np_predicted_equilibrial_values)))
				self.session.run(self.cfr_step_op, feed_dict={self.predicted_to_exp_values: np_predicted_cf_values})
				#print(np.count_nonzero(self.session.run(self.predicted_to_exp_values)))
				#self.session.run(self.cfr_step_op, {self.predicted_to_exp_values:tf.assign(self.predicted_to_exp_values, np_predicted_equilibrial_values)})
				#print(np.count_nonzero(self.session.run(self.expected_values[10])))
				# if verbose:
				# 	print("After:")
				# 	print_tensor(self.session, self.input_ranges)

				if step in register_strategies_on_step:
					self.average_strategies_over_steps["average_strategy_step{}".format(step)] = [
						self.session.run(strategy).tolist() for strategy in self.average_infoset_strategies[:self.trunk_depth]
					]
