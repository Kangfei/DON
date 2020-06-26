#!/bin/bash
# classification model steps = steps * n_iterations * n_trains
python2 train_rl_gorder.py \
 --batch_size 256 \
 --steps 200 \
 --learning_rate 0.0001 \
 --w 5 \
 --n_hidden 256 \
 --n_eval_data 1000 \
 --rl_learning_rate 0.0001 \
 --n_iterations 100 \
 --n_trains 3 \
 --tuning_rate 0.4 \
 --verbose True \
 --input_data_folder ../../data/wv \
 --model_dir rl_dnn_gorder_wv_2 \
  > result_rl_dnn_gorder_wv_2.txt 
