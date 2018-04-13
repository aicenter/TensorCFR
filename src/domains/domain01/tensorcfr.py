import tensorflow as tf

from src.domains.domain01.cfr_step import do_cfr_step
from src.domains.domain01.domain01 import cfr_step, infoset_strategies, acting_depth
from src.domains.domain01.uniform_strategies import get_infoset_uniform_strategies
from src.utils.tensor_utils import print_tensors

# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png


total_steps = 1000
cfr_step_op = do_cfr_step()
uniform_strategies = get_infoset_uniform_strategies()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print("Initializing strategies to uniform ones...\n")
	sess.run([tf.assign(ref=infoset_strategies[level], value=uniform_strategies[level]) for level in range(acting_depth)])
	print_tensors(sess, infoset_strategies)
	print("Running {} CFR+ iterations...\n".format(total_steps))
	for _ in range(total_steps):
		print("########## CFR+ step #{} ##########".format(cfr_step.eval()))
		sess.run(cfr_step_op)
		print_tensors(sess, infoset_strategies)
