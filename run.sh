#!/usr/bin/env bash
time=$(date "+%m_%d_%H:%M")
nohup python main.py \
            --dataset rest16 \
            --model_name_or_path /data1/SI-T2S/ABSA-QUAD/model_cache/cache_t5 \
            --n_gpu 7 \
            --do_pic \
            --do_train \
            --do_direct_eval \
            --use_prompt_flag \
            --use_sent_flag \
            --max_seq_length 1024 \
            --E_I ALL \
            --train_batch_size 2 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 2 \
            --learning_rate 2e-5 \
            --num_train_epochs 20 \
            >> ${time}__par_.log 2>&1 &