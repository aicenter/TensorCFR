from abc import abstractmethod
import tensorflow as tf


class AbstractNN:
	@abstractmethod
	def predict(self, input_features, pass_input_using=None):
		raise NotImplementedError

	#def __init__(self,threads):
		#self.graph = tf.Graph()
		#self.session = tf.Session(graph=self.graph,config=tf.ConfigProto(inter_op_parallelism_threads=threads,
		                                                                 #intra_op_parallelism_threads=threads))
	@abstractmethod
	def construct_saver(self):
		self.saver = tf.train.Saver()

	@abstractmethod
	def construct_checkpoint(self,steps):

		if self.saver in globals():

			try:

				self.checkpoint = tf.train.CheckpointSaverHook(saver=self.saver,save_steps=steps,checkpoint_basename="model.ckpt")

			except NameError:
				print("saver not defined yet")

	@abstractmethod
	def call_saver(self,path):

		if self.saver not in tf.global_variables():

			AbstractNN.construct_saver(self)

		self.saver.save(self.session,path)

	@abstractmethod
	def load_checkpoint(self,path):
		raise NotImplementedError



