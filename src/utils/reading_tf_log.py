import os
import tensorflow as tf

from src.commons.constants import PROJECT_ROOT

if __name__ == '__main__':
	# log_file_path = os.path.join(
	# 	PROJECT_ROOT,
	# 	'logs',
	# 	'Runner_CNN_IIGS6Lvl10_NPZ-2018-11-12_125341-bs=2,e=10,e=C-46,r=C-46,t=1',
	# 	'events.out.tfevents.1542023648.doom14.metacentrum.cz.v2')
	log_file_path = os.path.join(
		PROJECT_ROOT,
		'logs_KAREL',
		'Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_133801-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,C-46,C-46,C-46,C-46,C-46,C-46,r=C-138,C-138,C-138,C-138,C-138,C-138,C-138,t=1,tr=0.8',
		'events.out.tfevents.1541939902.konos1.fav.zcu.cz.v2')

	logs_values = dict()
	logs_steps = list()

	for e in tf.train.summary_iterator(log_file_path):

		# print(e.step)

		# logs_steps_len = len(logs_steps)
		#
		# if logs_steps_len == 0:
		# 	logs_steps.append(e.step)
		#
		# if logs_steps_len > 0 and e.step != logs_steps[logs_steps_len - 1]:
		# 	logs_steps.append(e.step)

		for v in e.summary.value:
			# print(str(v.tag) + " -> " + str(v.simple_value))
			if v.tag not in logs_values.keys():
				logs_values[v.tag] = {
					'steps': [e.step],
					'values': [v.simple_value]
				}
			else:
				logs_values[v.tag]['steps'].append(e.step)
				logs_values[v.tag]['values'].append(v.simple_value)
		# print("----")

	# print values
	for k in logs_values.keys():
		print(k)
		print('\t')
		print(logs_values[k]['steps'])
		print(logs_values[k]['values'])
		print("---")

	# plotting
	import matplotlib.pyplot as plt
	plotting_graph = 'train/loss'
	plt.plot(logs_values[plotting_graph]['steps'], logs_values[plotting_graph]['values'])
	plt.xlabel('Steps')
	plt.ylabel('L-infinity Error')
	plt.ylim(0, 0.2) # set the y axis to be shown in the interval (0, 0.2)
	plt.legend(['NN 1'])
	plt.title('Hello world')
	plt.show()
