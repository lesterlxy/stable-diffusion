"""make variations of input image"""

import argparse
import os
import time
from contextlib import nullcontext
from itertools import islice

import accelerate
import k_diffusion as K
import numpy as np
import PIL
import torch
import torch.nn as nn
from einops import rearrange, repeat
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from transformers import logging


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


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).half()
    return 2.0 * image - 1.0


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
    logging.set_verbosity_error()  # suppress BERT errors

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
        "-I", "--init-img", type=str, nargs="?", help="path to the input image"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples",
    )

    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
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
        "--fixed_code",
        action="store_true",
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
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

    accelerator = accelerate.Accelerator()

    opt = parser.parse_args()
    seed_everything(opt.seed)
    # seeds = torch.randint(-(2**63), 2**63 - 1, [accelerator.num_processes])
    # torch.manual_seed(seeds[accelerator.process_index].item())

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.half()
    model_wrap = K.external.CompVisDenoiser(model)
    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.kl:
        sampler = K.sampling.sample_lms
        print("Sampler set to k-lms")
    elif opt.kea:
        sampler = K.sampling.sample_euler_ancestral
        print("Sampler set to k-euler-ancestral")
    elif opt.kh:
        sampler = K.sampling.sample_heun
        print("Sampler set to k-heun")
    elif opt.kd:
        sampler = K.sampling.sample_dpm_2
        print("Sampler set to k-dpm-2")
    elif opt.kda:
        sampler = K.sampling.sample_dpm_2_ancestral
        print("Sampler set to k-dpm-2-ancestral")
    elif opt.ke:
        sampler = K.sampling.sample_euler
        print("Sampler set to k-euler")
    else:
        sampler = K.sampling.sample_euler
        print("Sampler set to default (k-euler)")

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

    assert os.path.isfile(opt.init_img)
    init_image = load_img(opt.init_img).to(device)
    init_image = repeat(init_image, "1 ... -> b ...", b=batch_size)
    init_latent = model.get_first_stage_encoding(
        model.encode_first_stage(init_image)
    )  # move to latent space

    assert 0.0 <= opt.strength <= 1.0, "can only work with strength in [0.0, 1.0]"
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")

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
                        model_wrap_cfg = CFGDenoiser(model_wrap)
                        extra_args = {"cond": c, "uncond": uc, "cond_scale": opt.scale}
                        # torch.manual_seed(opt.seed)  # changes manual seeding procedure
                        sigmas = model_wrap.get_sigmas(opt.ddim_steps)
                        noise = (
                            torch.randn_like(init_latent)
                            * sigmas[opt.ddim_steps - t_enc - 1]
                        )
                        init_noised = init_latent + noise
                        sigma_sched = sigmas[opt.ddim_steps - t_enc - 1 :]
                        samples = sampler(
                            model_wrap_cfg,
                            init_noised,
                            sigma_sched,
                            extra_args=extra_args,
                            disable=not accelerator.is_main_process,
                        )
                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp(
                            (x_samples + 1.0) / 2.0, min=0.0, max=1.0
                        )

                        if not opt.skip_save:
                            for x_sample in x_samples:
                                x_sample = 255.0 * rearrange(
                                    x_sample.cpu().numpy(), "c h w -> h w c"
                                )
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f"{base_count:05}.png")
                                )
                                base_count += 1
                        all_samples.append(x_samples)

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
