import argparse
import os
from contextlib import nullcontext
from itertools import islice

import accelerate
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-P",
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "-N",
        "--steps",
        type=int,
        default=50,
        help="number of sampling steps",
    )
    parser.add_argument(
        "--kl",
        action="store_true",
        help="use k-lms sampling",
    )
    parser.add_argument(
        "--ke",
        action="store_true",
        help="use k-euler sampling (default)",
    )
    parser.add_argument(
        "--kea",
        action="store_true",
        help="use k-euler-ancestral sampling",
    )
    parser.add_argument(
        "--kh",
        action="store_true",
        help="use k-heun sampling",
    )
    parser.add_argument(
        "--kd",
        action="store_true",
        help="use k-dpm-2 sampling",
    )
    parser.add_argument(
        "--kda",
        action="store_true",
        help="use k-dpm-2-ancestral sampling",
    )
    parser.add_argument(
        "--leaked",
        action="store_true",
        help="uses the leaked v1.3 model",
    )
    parser.add_argument(
        "--fixed_code",
        action="store_true",
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "-H",
        "--height",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "-W",
        "--width",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--square",
        action="store_true",
        help="size preset",
    )
    parser.add_argument(
        "--portrait",
        action="store_true",
        help="size preset",
    )
    parser.add_argument(
        "--landscape",
        action="store_true",
        help="size preset",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "-C",
        "--scale",
        type=float,
        default=7.5,
        help="cfg scale - unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1-4/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=62353535,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast",
    )
    parser.add_argument(
        "--four",
        action="store_true",
        help="grid",
    )
    parser.add_argument(
        "--six",
        action="store_true",
        help="grid",
    )
    parser.add_argument(
        "--nine",
        action="store_true",
        help="grid",
    )

    opt = parser.parse_args()

    if opt.square:
        opt.height = 640
        opt.width = 640
    elif opt.portrait:
        opt.height = 768
        opt.width = 576
    elif opt.landscape:
        opt.height = 576
        opt.width = 768

    if opt.four:
        opt.n_iter = 4
        opt.n_rows = 2
    elif opt.six:
        opt.n_iter = 6
        opt.n_rows = 3
    elif opt.nine:
        opt.n_iter = 9
        opt.n_rows = 3

    if opt.leaked:
        print("Falling back to leaked v1.3 model...")
        opt.ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"

    seed_everything(opt.seed)

    accelerator = accelerate.Accelerator()

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.half()
    model_wrap = K.external.CompVisDenoiser(model)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    if opt.kl:
        sampler = K.sampling.sample_lms
    elif opt.kea:
        sampler = K.sampling.sample_euler_ancestral
    elif opt.kh:
        sampler = K.sampling.sample_heun
    elif opt.kd:
        sampler = K.sampling.sample_dpm_2
    elif opt.kda:
        sampler = K.sampling.sample_dpm_2_ancestral
    else:
        sampler = K.sampling.sample_euler

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.channels, opt.height // opt.f, opt.width // opt.f]
                        sigmas = model_wrap.get_sigmas(opt.steps)
                        x = (
                            torch.randn([opt.n_samples, *shape], device=device)
                            * sigmas[0]
                        )  # for GPU draw
                        model_wrap_cfg = CFGDenoiser(model_wrap)
                        extra_args = {"cond": c, "uncond": uc, "cond_scale": opt.scale}
                        samples_ddim = sampler(
                            model_wrap_cfg,
                            x,
                            sigmas,
                            extra_args=extra_args,
                            disable=not accelerator.is_main_process,
                        )
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                        )

                        if not opt.skip_save:
                            for x_sample in x_samples_ddim:
                                x_sample = 255.0 * rearrange(
                                    x_sample.cpu().numpy(), "c h w -> h w c"
                                )
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f"{base_count:05}.png")
                                )
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_samples_ddim)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, "n b c h w -> (n b) c h w")
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(
                        os.path.join(outpath, f"grid-{grid_count:04}.png")
                    )
                    grid_count += 1

    print(
        f"Your samples are ready and waiting for you here: \n{outpath} \n" f" \nEnjoy."
    )


if __name__ == "__main__":
    main()
