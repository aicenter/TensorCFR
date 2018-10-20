for i in $(seq 7)
do
	qsub metacentrum_${i}_layers.sh
	#CMD="qsub metacentrum_$i_layers.sh"
	#echo $CMD
	#$CMD
done
