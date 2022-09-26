#!/bin/sh

gpu_id=10,11,12,15,0,5,6

continue_from=

if [ -z ${continue_from} ]; then
	log_name='avDprnn_'$(date '+%d-%m-%Y(%H:%M:%S)')
	mkdir -p logs/$log_name
else
	log_name=${continue_from}
fi

CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=6411 \
main.py \
--log_name $log_name \
\
--audio_direc '/home/panzexu/datasets/voxceleb2/audio_clean/' \
--visual_direc '/home/panzexu/datasets/voxceleb2/visual_embedding/lip/' \
--mix_lst_path '/home/panzexu/datasets/voxceleb2/audio_mixture/2_mix_min_pretrain/mixture_data_list_2mix.csv' \
--mixture_direc '/home/panzexu/datasets/voxceleb2/audio_mixture/2_mix_min_pretrain/' \
--C 2 \
\
--effec_batch_size 48 \
--accu_grad 1 \
--batch_size 12 \
\
\
--epochs 200 \
--use_tensorboard 1 \
>logs/$log_name/console.txt 2>&1

# --continue_from ${continue_from} \