#!/bin/bash

python run_benchmark.py \
    --task "orient" \
    --model "sd-turbo" \
    --save_dir /root/data/orientgen_bench/experiments \
    --n_iters 100 \
    --orient_weighting 1.0 \
    --save_last \
    --cache_dir "/root/data/model/" \
    --disable_clip --disable_imagereward --disable_pickscore --disable_aesthetic --disable_hps \
    --lr 2.0 \
    --benchmark_path "/root/data/orientgen_bench/data_small.json" \
    --noise_optimize --n_noises 4 \
    --seed 0

python run_benchmark.py \
    --task "orient" \
    --model "sd-turbo" \
    --save_dir /root/data/orientgen_bench/experiments \
    --n_iters 100 \
    --orient_weighting 1.0 \
    --save_last \
    --cache_dir "/root/data/model/" \
    --disable_clip --disable_imagereward --disable_pickscore --disable_aesthetic --disable_hps \
    --lr 2.0 \
    --benchmark_path "/root/data/orientgen_bench/data_small.json" \
    --noise_optimize --n_noises 10 \
    --seed 0


python run_benchmark.py \
    --task "orient" \
    --model "sd-turbo" \
    --save_dir /root/data/orientgen_bench/experiments \
    --n_iters 100 \
    --orient_weighting 1.0 \
    --save_last \
    --cache_dir "/root/data/model/" \
    --disable_clip --disable_imagereward --disable_pickscore --disable_aesthetic --disable_hps \
    --lr 2.0 \
    --benchmark_path "/root/data/orientgen_bench/data_small.json" \
    --noise_optimize --n_noises 20 \
    --seed 0

