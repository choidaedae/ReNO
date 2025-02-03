#!/bin/bash

for seed in {0..99}
do
    python run_orient.py \
        --task "orient" \
        --model "sd-turbo" \
        --save_all_images \
        --n_iters 100 \
        --orient_weighting 1.0 \
        --lr 1.0 \
        --seed $seed
done

#python run_orient.py  --task "orient" --model "sd-turbo" --save_all_images --n_iters 100 --orient_weighting 1.0 --lr 0.1

#python run_orient.py  --task "orient" --model "sd-turbo" --save_all_images --n_iters 100 --orient_weighting 1.0 --lr 0.1 --disable_reg

#python run_orient.py  --task "orient" --model "sd-turbo" --save_all_images --n_iters 100 --orient_weighting 1.0 --lr 0.5

#python run_orient.py  --task "orient" --model "sd-turbo" --save_all_images --n_iters 100 --orient_weighting 1.0 --lr 0.5 --disable_reg

#python run_orient.py  --task "orient" --model "sd-turbo" --save_all_images --n_iters 100 --orient_weighting 1.0 --lr 1.0

#python run_orient.py  --task "orient" --model "sd-turbo" --save_all_images --n_iters 100 --orient_weighting 1.0 --lr 1.0 --disable_reg
