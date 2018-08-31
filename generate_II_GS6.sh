size=${1:-1}
datasets=${2:-1000}
for dataset in $(seq 0 $datasets);
do
	echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	seed=$((dataset * size))
	echo "Generation #$dataset, starting with dataset_seed $seed:"
	CUDA_VISIBLE_DEVICES=0 python3.6 -m src.nn.data.generate_data_of_IIGS6 --seed $seed --size $size
	echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
done
