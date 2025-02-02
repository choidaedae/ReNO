import json
import logging
import os

import blobfile as bf
import torch
from datasets import load_dataset
from tqdm import tqdm

from arguments import parse_args
from models import get_model, get_multi_apply_fn
from rewards import get_reward_losses
from training import LatentNoiseTrainer, get_optimizer
import numpy as np

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    seed_everything(args.seed)
    bf.makedirs(f"{args.save_dir}/logs/{args.task}")
    # Set up logging and name settings
    logger = logging.getLogger()
    settings = (
        f"{args.model}{'_' + args.prompt if args.task == 't2i-compbench' else ''}"
        f"{'_no-optim' if args.no_optim else ''}_{args.seed if args.task != 'geneval' else ''}"
        f"_lr{args.lr}_gc{args.grad_clip}_iter{args.n_iters}"
        f"_reg{args.reg_weight if args.enable_reg else '0'}"
        f"{'_pickscore' + str(args.pickscore_weighting) if args.enable_pickscore else ''}"
        f"{'_clip' + str(args.clip_weighting) if args.enable_clip else ''}"
        f"{'_hps' + str(args.hps_weighting) if args.enable_hps else ''}"
        f"{'_imagereward' + str(args.imagereward_weighting) if args.enable_imagereward else ''}"
        f"{'_aesthetic' + str(args.aesthetic_weighting) if args.enable_aesthetic else ''}"
    )
    file_stream = open(f"{args.save_dir}/logs/{args.task}/{settings}.txt", "w")
    handler = logging.StreamHandler(file_stream)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel("INFO")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logging.info(args)
    if args.device_id is not None:
        logging.info(f"Using CUDA device {args.device_id}")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = torch.device("cuda")
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    # Get reward losses
    reward_losses = get_reward_losses(args, dtype, device, args.cache_dir)

    # Get model and noise trainer
    pipe = get_model(
        args.model, dtype, device, args.cache_dir, args.memsave, args.cpu_offloading
    )
    trainer = LatentNoiseTrainer(
        reward_losses=reward_losses,
        model=pipe,
        n_iters=args.n_iters,
        n_inference_steps=args.n_inference_steps,
        seed=args.seed,
        save_all_images=args.save_all_images,
        device=device,
        no_optim=args.no_optim,
        regularize=args.enable_reg,
        regularization_weight=args.reg_weight,
        grad_clip=args.grad_clip,
        log_metrics=args.task == "single" or not args.no_optim,
        imageselect=args.imageselect,
    )

    # Create latents
    if args.model == "flux":
        # currently only support 512x512 generation
        shape = (1, 16 * 64, 64)
    elif args.model != "pixart":
        height = pipe.unet.config.sample_size * pipe.vae_scale_factor
        width = pipe.unet.config.sample_size * pipe.vae_scale_factor
        shape = (
            1,
            pipe.unet.in_channels,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        )
    else:
        height = pipe.transformer.config.sample_size * pipe.vae_scale_factor
        width = pipe.transformer.config.sample_size * pipe.vae_scale_factor
        shape = (
            1,
            pipe.transformer.config.in_channels,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        )
    enable_grad = not args.no_optim
    if args.enable_multi_apply:
        multi_apply_fn = get_multi_apply_fn(
            model_type=args.multi_step_model,
            seed=args.seed,
            pipe=pipe,
            cache_dir=args.cache_dir,
            device=device,
            dtype=dtype,
        )
    else:
        multi_apply_fn = None

    if args.task == "orient":
        prompt_list_file = "./assets/orient_prompts.json"
        with open(prompt_list_file) as fp:
            metadatas = json.load(fp)["data"]
        init_latents = torch.randn(shape, device=device, dtype=dtype).clone()
        for index, metadata in enumerate(metadatas):
            # Get new latents and optimizer
            latents = torch.nn.Parameter(init_latents, requires_grad=True)
            optimizer = get_optimizer(args.optim, latents, args.lr, args.nesterov)
            prompt = metadata["prompt"]
            orientation = np.array(metadata["orientation"])
            save_dir = f"{args.save_dir}/{args.task}/{prompt}_({orientation[0]},{orientation[1]},{orientation[2]})_reg_{args.enable_reg}_lr_{args.lr}_seed_{args.seed}"
            os.makedirs(save_dir, exist_ok=True)    
            init_image, best_image, init_rewards, best_rewards = trainer.train_orient(
                latents, prompt, orientation, optimizer, save_dir, multi_apply_fn
            )
            logging.info(f"Initial rewards: {init_rewards}")
            logging.info(f"Best rewards: {best_rewards}")
            if index == 0:
                total_best_rewards = {k: 0.0 for k in best_rewards.keys()}
                total_init_rewards = {k: 0.0 for k in best_rewards.keys()}
            for k in best_rewards.keys():
                total_best_rewards[k] += best_rewards[k]
                total_init_rewards[k] += init_rewards[k]
    else:
        raise ValueError(f"Unknown task {args.task}")
    # log total rewards
    logging.info(f"Mean initial rewards: {total_init_rewards}")
    logging.info(f"Mean best rewards: {total_best_rewards}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
