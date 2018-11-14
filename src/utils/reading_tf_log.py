import os
import tensorflow as tf

from src.commons.constants import PROJECT_ROOT

if __name__ == '__main__':
	log_file_path = os.path.join(
		PROJECT_ROOT,
		'logs',
		'Runner_CNN_IIGS6Lvl10_NPZ-2018-11-12_125341-bs=2,e=10,e=C-46,r=C-46,t=1',
		'events.out.tfevents.1542023648.doom14.metacentrum.cz.v2')

	logs_values = dict()
	logs_steps = list()

	for e in tf.train.summary_iterator(log_file_path):

		print(e.step)

		# logs_steps_len = len(logs_steps)
		#
		# if logs_steps_len == 0:
		# 	logs_steps.append(e.step)
		#
		# if logs_steps_len > 0 and e.step != logs_steps[logs_steps_len - 1]:
		# 	logs_steps.append(e.step)

		for v in e.summary.value:
			print(str(v.tag) + " -> " + str(v.simple_value))
			if v.tag not in logs_values.keys():
				logs_values[v.tag] = {
					'steps': [e.step],
					'values': [v.simple_value]
				}
			else:
				logs_values[v.tag]['steps'].append(e.step)
				logs_values[v.tag]['values'].append(v.simple_value)
		print("----")


	for k in logs_values.keys():
		print(k)
		print('\t')
		print(logs_values[k]['steps'])
		print(logs_values[k]['values'])
		print("---")
