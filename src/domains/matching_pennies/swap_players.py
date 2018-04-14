import tensorflow as tf

from src.constants import PLAYER1, PLAYER2
from src.domains.matching_pennies.domain_definitions import cfr_step, \
	current_updating_player, current_opponent, print_player_variables


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

def get_the_other_player_of(tensor_variable_of_player):
	# for 2-player games with players of indices 1 and 2, the other players are 3-1=2 and 3-2=1
	return 3 - tensor_variable_of_player


def swap_players():
	return tf.group(
			[
				current_updating_player.assign(get_the_other_player_of(current_updating_player)),
				current_opponent.assign(get_the_other_player_of(current_opponent)),
			],
			name="swap_players",
	)


if __name__ == '__main__':
	increment_cfr_step_op = tf.assign_add(ref=cfr_step, value=1, name="increment_cfr_step_op")
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("PLAYER1 is {}, PLAYER2 is {}".format(PLAYER1, PLAYER2))
		for _ in range(5):
			print("########## CFR step {} ##########\n".format(cfr_step.eval()))
			print_player_variables(session=sess)
			sess.run([swap_players(), increment_cfr_step_op])
