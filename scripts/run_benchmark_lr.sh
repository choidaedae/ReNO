#!/bin/bash

python run_run_benchmark.py \
    --task "orient" \
    --model "sd-turbo" \
    --save_dir /root/data/orientgen_bench/ \
    --n_iters 100 \
    --orient_weighting 1.0 \
    --save_last \
    --cache_dir "/root/data/model/" \
    --disable_clip --disable_imagereward --disable_pickscore --disable_aesthetic --disable_hps \
    --lr 1.0 \
    --benchmark_path "/root/data/orientgen_bench/data/data_small.json" \
    --seed 0

python run_benchmark.py \
    --task "orient" \
    --model "sd-turbo" \
    --save_dir /root/data/orientgen_bench/ \
    --n_iters 100 \
    --orient_weighting 1.0 \
    --save_last \
    --cache_dir "/root/data/model/" \
    --disable_clip --disable_imagereward --disable_pickscore --disable_aesthetic --disable_hps \
    --lr 3.0 \
    --benchmark_path "/root/data/orientgen_bench/data/data_small.json" \
    --seed 0


python run_benchmark.py \
    --task "orient" \
    --model "sd-turbo" \
    --save_dir /root/data/orientgen_bench/ \
    --n_iters 100 \
    --orient_weighting 1.0 \
    --save_last \
    --cache_dir "/root/data/model/" \
    --disable_clip --disable_imagereward --disable_pickscore --disable_aesthetic --disable_hps \
    --lr 5.0 \
    --benchmark_path "/root/data/orientgen_bench/data/data_small.json" \
    --seed 0

