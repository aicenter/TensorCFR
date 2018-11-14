import os
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

from src.commons.constants import PROJECT_ROOT

if __name__ == '__main__':
	# log_file_path1 = os.path.join(
	# 	PROJECT_ROOT,
	# 	'logs',
	# 	'Runner_CNN_IIGS6Lvl10_NPZ-2018-11-12_125341-bs=2,e=10,e=C-46,r=C-46,t=1',
	# 	'events.out.tfevents.1542023648.doom14.metacentrum.cz.v2')
	log_file_path1 = os.path.join(
		PROJECT_ROOT,
		'logs_KAREL',
		'Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_021744-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,C-46,C-46,r=C-138,C-138,C-138,t=1,tr=0.8',
		'events.out.tfevents.1541899082.glados10.cerit-sc.cz.v2')
	log_file_path2 = os.path.join(
		PROJECT_ROOT,
		'logs_KAREL',
		'Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_021752-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,C-46,r=C-138,C-138,t=1,tr=0.8',
		'events.out.tfevents.1541899086.white1.cerit-sc.cz.v2')
	log_file_path3 = os.path.join(
		PROJECT_ROOT,
		'logs_KAREL',
		'Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_021836-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,r=C-138,t=1,tr=0.8',
		'events.out.tfevents.1541899127.white1.cerit-sc.cz.v2')
	log_file_path4 = os.path.join(
		PROJECT_ROOT,
		'logs_KAREL',
		'Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_030427-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,C-46,C-46,C-46,C-46,r=C-138,C-138,C-138,C-138,C-138,t=1,tr=0.8',
		'events.out.tfevents.1541901887.glados16.cerit-sc.cz.v2')
	log_file_path5 = os.path.join(
		PROJECT_ROOT,
		'logs_KAREL',
		'Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_133801-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,C-46,C-46,C-46,C-46,C-46,C-46,r=C-138,C-138,C-138,C-138,C-138,C-138,C-138,t=1,tr=0.8',
		'events.out.tfevents.1541939902.konos1.fav.zcu.cz.v2')
	log_files_paths = [
		log_file_path1,
		log_file_path2,
		log_file_path3,
		log_file_path4,
		log_file_path5
	]

	plotting_values = 'train/loss'

	plot_names = ['1', '2', '3', '4', '5']

	graph_logs_values = list()

	for log_file_path in log_files_paths:
		logs_values = dict()

		for e in tf.train.summary_iterator(log_file_path):
			for v in e.summary.value:
				if v.tag not in logs_values.keys():
					logs_values[v.tag] = {
						'steps': [e.step],
						'values': [v.simple_value]
					}
				else:
					logs_values[v.tag]['steps'].append(e.step)
					logs_values[v.tag]['values'].append(v.simple_value)
		graph_logs_values.append(logs_values)

	for idx, graph_log_values in enumerate(graph_logs_values):
		plt.plot(graph_log_values[plotting_values]['steps'], graph_log_values[plotting_values]['values'], linewidth=1)

	# # print values
	# for k in logs_values.keys():
	# 	print(k)
	# 	print('\t')
	# 	print(logs_values[k]['steps'])
	# 	print(logs_values[k]['values'])
	# 	print("---")
	#
	# # plotting
	# import matplotlib.pyplot as plt
	# plotting_graph = 'train/loss'
	# plt.plot(logs_values[plotting_graph]['steps'], logs_values[plotting_graph]['values'])
	#
	# plotting_graph = 'train/l_infinity_error'
	# plt.plot(logs_values[plotting_graph]['steps'], [0.150]*len(logs_values[plotting_graph]['steps']))
	#
	#
	plt.xlabel('Steps')
	plt.ylabel('ylabel')
	plt.ylim(0.025, 0.2)# 0.08, 0.11) # set the y axis to be shown in the fixed interval (0, 0.2)
	plt.legend(plot_names)
	plt.title('title')
	plt.savefig("pic.png")
	# plt.plot()
