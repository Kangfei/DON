#!/bin/bash
python2 train_dnn_gorder.py --batch_size 64 --steps 90000 --learning_rate 0.0001 --verbose False  --input_data_folder ../data/wv --w 5 --model_dir dnn_models_wv_gorder_3/ > result_wv_gorder_3.txt
