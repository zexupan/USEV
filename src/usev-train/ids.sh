#!/bin/sh

gpu_id=6,8

continue_from=

if [ -z ${continue_from} ]; then
	log_name='avDprnn_'$(date '+%d-%m-%Y(%H:%M:%S)')
	mkdir logs/$log_name
else
	log_name=${continue_from}
fi

CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=2 \
--master_port=3235 \
main.py \
--log_name $log_name \
\
--audio_direc '/home/panzexu/datasets/iemocap/uss/audio_clean/' \
--visual_direc '/home/panzexu/datasets/iemocap/uss/visual_embedding/lip/' \
--mix_lst_path '/home/panzexu/datasets/iemocap/uss/audio_mixture/2_mix_sparse_noise/mixture_data_list_mix.csv' \
--mixture_direc '/home/panzexu/datasets/iemocap/uss/audio_mixture/2_mix_sparse_noise/' \
--C 2 \
--max_length 6 \
\
--effec_batch_size 16 \
--accu_grad 1 \
--batch_size 4 \
\
--epochs 200 \
\
--w_a 0.005 \
--w_b 1.0 \
--w_c 1.0 \
--w_d 0.005 \
\
--use_tensorboard 1 \
>logs/$log_name/console.txt 2>&1

# --continue_from ${continue_from} \
# effec batch 96
