import json
import os

import blobfile as bf
import torch
from datasets import load_dataset
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

from arguments import parse_args
from models import get_model, get_multi_apply_fn
from rewards import get_reward_losses
from training import LatentNoiseTrainer, get_optimizer
import numpy as np

from pathlib import Path
import yaml
import random

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    seed_everything(args.seed)
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
    if args.device_id is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = torch.device("cuda")
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    # Get reward losses
    reward_losses = get_reward_losses(args, dtype, device, args.cache_dir)

    def save_config(path, dic):
        if isinstance(path, str):
            path = Path(path)
        if path.suffix == "":
            path.mkdir(exist_ok=True, parents=True)
            path = path / "config.yaml"
        elif path.suffix == ".yaml" or path.suffix == ".yml":
            dirpath = path.parent
            dirpath.mkdir(exist_ok=True, parents=True)
        else:
            raise ValueError
        with open(path, "w") as f:
            yaml.dump(dic, f)

    args_dic = vars(args)
    args.save_dir = f"{args.save_dir}/{args.model}"
    save_config(args.save_dir + "/config.yaml", args_dic)

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
        with open(args.benchmark_path) as fp:
            metadatas = json.load(fp)
            prompts = metadatas["prompts"] 
            orientations = metadatas["orientations"]
        if args.noise_optimize:
            seeds = np.random.randint(0, 10000, size=args.n_noises-1).tolist()
            generator = [torch.Generator("cuda").manual_seed(args.seed)] + [torch.Generator("cuda").manual_seed(seed) for seed in seeds]
            init_latents = [torch.randn(shape, generator=generator[i], device=device, dtype=dtype) for i in range(args.n_noises)]
            init_latents = torch.cat(init_latents, dim=0).to(device)
        else:
            init_latents = torch.randn(shape, generator=torch.Generator("cuda").manual_seed(args.seed), device=device, dtype=dtype)
        for i, prompt in enumerate(prompts):
            for j, orientation in enumerate(orientations):
                if args.noise_optimize:
                    losses = torch.zeros(init_latents.shape[0])
                    for k in range(init_latents.shape[0]):
                        image = trainer.model.apply(
                            latents=init_latents[k].unsqueeze(0).clone(),
                            prompt=prompt,
                            generator = generator[0],
                            num_inference_steps = args.n_inference_steps,
                        )
                        losses[k] = trainer.reward_losses[-1](image, orientation)
                    best_noise_idx = int(torch.argmin(losses))
                    latents = init_latents[best_noise_idx]
                    latents = torch.nn.Parameter(latents.unsqueeze(0).clone(), requires_grad=True)
                else:
                    latents = torch.nn.Parameter(init_latents.clone(), requires_grad=True)
                optimizer = get_optimizer(args.optim, latents, args.lr, args.nesterov)
                save_dir = f"{args.save_dir}/reg_{args.enable_reg}_lr_{args.lr}_seed_{args.seed}_noise_optimize_{args.noise_optimize}_noises_{args.n_noises}"
                os.makedirs(save_dir, exist_ok=True)    
                init_image, last_image, _, _= trainer.train_orient(
                    latents, prompt, orientation, optimizer, save_dir, multi_apply_fn, args.save_last, args.noise_optimize
                )
                init_image.save(f"{save_dir}/prompt_{i}_orientation_{j}_init.png")
                last_image.save(f"{save_dir}/prompt_{i}_orientation_{j}_result.png")

        with open(f"{save_dir}/benchmark.json", "w") as f:
            json.dump(metadatas, f)
    else:
        raise ValueError(f"Unknown task {args.task}")
    # log total rewards


if __name__ == "__main__":
    args = parse_args()
    main(args)
