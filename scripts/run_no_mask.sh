
for SEED in {0..9}; do
    python run_benchmark.py \
        --task "orient" \
        --model "sd-turbo" \
        --save_dir ./results/masking_test/no_mask/ \
        --n_iters 100 \
        --orient_weighting 1.0 \
        --save_last --save_all_images \
        --cache_dir "/root/data/model/" \
        --disable_clip --disable_imagereward --disable_pickscore --disable_aesthetic --disable_hps \
        --lr 3.0 --background_preprocess "no" \
        --benchmark_path "/root/data/orientgen_bench/data_for_masking.json" \
        --seed $SEED
    
    python run_benchmark.py \
        --task "orient" \
        --model "sd-turbo" \
        --save_dir ./results/masking_test/no_mask/ \
        --n_iters 100 \
        --orient_weighting 1.0 \
        --save_last  --save_all_images \
        --cache_dir "/root/data/model/" \
        --disable_clip --disable_imagereward --disable_pickscore --disable_aesthetic --disable_hps \
        --lr 1.0 --background_preprocess "no" \
        --benchmark_path "/root/data/orientgen_bench/data_for_masking.json" \
        --seed $SEED

    python run_benchmark.py \
        --task "orient" \
        --model "sd-turbo" \
        --save_dir ./results/masking_test/no_mask/ \
        --n_iters 100 \
        --orient_weighting 1.0 \
        --save_last --save_all_images \
        --cache_dir "/root/data/model/" \
        --disable_clip --disable_imagereward --disable_pickscore --disable_aesthetic --disable_hps \
        --lr 2.0 --background_preprocess "no" \
        --benchmark_path "/root/data/orientgen_bench/data_for_masking.json" \
        --seed $SEED
    done

