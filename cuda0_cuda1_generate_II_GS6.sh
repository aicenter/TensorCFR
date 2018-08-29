cmd='python3.6 -m src.nn.data.generate_data_of_IIGS6'
num_launches=1
num_gpus=2
for l in $(seq 1 $num_launches);
do
	echo "######################################## Launch #$l started ########################################"
	for gpu_id in $(seq 0 $num_gpus); 	# TODO fix exclusive interval
	do
		echo "> Launching process #$gpu_id" && echo CUDA_VISIBLE_DEVICES=$gpu_id $cmd &
	done
	wait
	echo "########################################  Launch #$l ended  ########################################"
	echo ' '
done
