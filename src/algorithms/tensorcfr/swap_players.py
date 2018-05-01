import tensorflow as tf

from src.commons.constants import PLAYER1, PLAYER2


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

def get_the_other_player_of(tensor_variable_of_player):
	with tf.variable_scope("get_the_other_player"):
		return tf.where(
				condition=tf.equal(tensor_variable_of_player, PLAYER1),
				x=PLAYER2,
				y=PLAYER1,
				name="get_the_other_player"
		)


def swap_players():
	from src.commons.constants import INT_DTYPE
	with tf.variable_scope("domain_definitions", reuse=True):
		updating_player = tf.get_variable("current_updating_player", dtype=INT_DTYPE)
		opponent = tf.get_variable("current_opponent", dtype=INT_DTYPE)
	with tf.variable_scope("swap_players"):
		with tf.variable_scope("new_updating_player"):
			assign_new_updating_player = tf.assign(
					ref=updating_player,
					value=get_the_other_player_of(updating_player),
					name="assign_new_updating_player",
			)
		with tf.variable_scope("new_opponent"):
			assign_opponent = tf.assign(
					ref=opponent,
					value=get_the_other_player_of(opponent),
					name="assign_new_opponent",
			)
		return tf.tuple(
				[
					assign_new_updating_player,
					assign_opponent,
				],
				name="swap",
		)


if __name__ == '__main__':
	from src.domains.domain01.domain_definitions import cfr_step, print_misc_variables

	increment_cfr_step_op = tf.assign_add(ref=cfr_step, value=1, name="increment_cfr_step_op")
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("PLAYER1 is {}, PLAYER2 is {}".format(PLAYER1, PLAYER2))
		for _ in range(5):
			print("########## CFR step {} ##########\n".format(cfr_step.eval()))
			print_misc_variables(session=sess)
			sess.run([swap_players(), increment_cfr_step_op])
