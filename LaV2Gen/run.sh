#!/usr/bin/env bash


# GPT2 (Text)
# ** Train **
python main.py \
--mode train \
--expt_dir ./results \
--expt_name ATM_txt \
--text T \
--model gpt2 \
--max_text 32 \
--data_dir C:/Users/acer/Downloads/projects_symbolic-knowledge-decoding_ATOMIC10X.jsonl \
--run bs_8 \
--batch 128 \
--gpu 0 \
--num_work 1