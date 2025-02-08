#!/bin/bash

python run_benchmark.py \
    --task "orient" \
    --model "sd-turbo" \
    --save_dir ./results/resampled \
    --n_iters 50 \
    --orient_weighting 1.0 \
    --save_last \
    --cache_dir "/root/data/model/" \
    --disable_clip --disable_imagereward --disable_pickscore --disable_aesthetic --disable_hps \
    --lr 3.0 \
    --benchmark_path "/root/data/orientgen_bench/data_car.json" \
    --save_all --sigmas 20.0 2.0 1.0 \
    --seed 0

python run_benchmark.py \
    --task "orient" \
    --model "sd-turbo" \
    --save_dir ./results/var20_car \
    --n_iters 50 \
    --orient_weighting 1.0 \
    --save_last \
    --cache_dir "/root/data/model/" \
    --disable_clip --disable_imagereward --disable_pickscore --disable_aesthetic --disable_hps \
    --lr 1.0 \
    --benchmark_path "/root/data/orientgen_bench/data_car.json" \
    --save_all --sigmas 20.0 2.0 1.0 \
    --seed 0

python run_benchmark.py \
    --task "orient" \
    --model "sd-turbo" \
    --save_dir ./results/var20_car \
    --n_iters 50 \
    --orient_weighting 1.0 \
    --save_last \
    --cache_dir "/root/data/model/" \
    --disable_clip --disable_imagereward --disable_pickscore --disable_aesthetic --disable_hps \
    --lr 5.0 \
    --benchmark_path "/root/data/orientgen_bench/data_car.json" \
    --save_all --sigmas 20.0 2.0 1.0 \
    --seed 0

