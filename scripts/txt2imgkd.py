import argparse
import gc
import os
import random
import sys
from contextlib import nullcontext
from itertools import islice

import accelerate
import clip
import k_diffusion as K
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from einops import rearrange
from gfpgan import GFPGANer
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from transformers import logging
from src.blip.models.blip import blip_decoder
from torchvision.transforms.functional import InterpolationMode

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
GFPGAN_dir = "./src/GFPGAN"


def load_GFPGAN():
    model_name = "GFPGANv1.3"
    model_path = os.path.join(GFPGAN_dir, "experiments/pretrained_models", model_name + ".pth")
    if not os.path.isfile(model_path):
        raise Exception("GFPGAN model not found at path " + model_path)

    sys.path.append(os.path.abspath(GFPGAN_dir))

    return GFPGANer(model_path=model_path, upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None)


def split_weighted_subprompts(text):
    """
    grabs all text up to the first occurrence of ':'
    uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
    if ':' has no value defined, defaults to 1.0
    repeats until no text remaining
    """
    remaining = len(text)
    prompts = []
    weights = []
    while remaining > 0:
        if ":" in text:
            idx = text.index(":")  # first occurrence from start
            # grab up to index as sub-prompt
            prompt = text[:idx]
            remaining -= idx
            # remove from main text
            text = text[idx + 1 :]
            # find value for weight
            if " " in text:
                idx = text.index(" ")  # first occurence
            else:  # no space, read to end
                idx = len(text)
            if idx != 0:
                try:
                    weight = float(text[:idx])
                except:  # couldn't treat as float
                    print(f"Warning: '{text[:idx]}' is not a value, are you missing a space?")
                    weight = 1.0
            else:  # no value found
                weight = 1.0
            # remove from main text
            remaining -= idx
            text = text[idx + 1 :]
            # append the sub-prompt and its weight
            prompts.append(prompt)
            weights.append(weight)
        else:  # no : found
            if len(text) > 0:  # there is still text though
                # take remainder as weight 1
                prompts.append(text)
                weights.append(1.0)
            remaining = 0
    return prompts, weights


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_img(path, target_w, target_h):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    if w != target_w or h != target_h:
        print(f"mismatch with target size ({target_w}, {target_h}), resizing")
        image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        w, h = image.size
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).half()
    return 2.0 * image - 1.0


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


class Interrogator:
    # credit to https://colab.research.google.com/github/pharmapsychotic/clip-interrogator/blob/main/clip_interrogator.ipynb
    def __init__(self, device, blip_image_eval_size: int = 384):
        def load_list(filename):
            with open(filename, "r", encoding="utf-8", errors="replace") as f:
                items = [line.strip() for line in f.readlines()]
            return items

        self.device = device

        # "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model*_base_caption.pth"
        blip_model_url = os.path.join("models", "blip", "model_base_caption.pth")
        blip_model_url = os.path.join(os.getcwd(), blip_model_url)
        self.blip_image_eval_size = blip_image_eval_size
        self.blip_model = blip_decoder(pretrained=blip_model_url, image_size=blip_image_eval_size, vit="base", med_config="src/blip/configs/med_config.json")
        self.blip_model.eval()
        self.blip_model.half()
        self.blip_model.to(device)

        data_path = "./src/clip-interrogator/data/"
        self.artists = load_list(os.path.join(data_path, "artists.txt"))
        self.flavors = load_list(os.path.join(data_path, "flavors.txt"))
        self.mediums = load_list(os.path.join(data_path, "mediums.txt"))
        self.movements = load_list(os.path.join(data_path, "movements.txt"))

        sites = [
            "Artstation",
            "behance",
            "cg society",
            "cgsociety",
            "deviantart",
            "dribble",
            "flickr",
            "instagram",
            "pexels",
            "pinterest",
            "pixabay",
            "pixiv",
            "polycount",
            "reddit",
            "shutterstock",
            "tumblr",
            "unsplash",
            "zbrush central",
        ]
        self.trending_list = [site for site in sites]
        self.trending_list.extend(["trending on " + site for site in sites])
        self.trending_list.extend(["featured on " + site for site in sites])
        self.trending_list.extend([site + " contest winner" for site in sites])

        self.model_name = "ViT-L/14"  # only one model for stable diffusion as recommended by pharmapsychotic/clip-interrogator
        self.model, self.preprocess = clip.load(self.model_name)
        self.model.eval()
        self.model.to(device)

    def __del__(self):
        del self.model
        del self.blip_model
        torch.cuda.empty_cache()
        gc.collect()

    def rank(self, image_features, text_array, top_count=1):
        top_count = min(top_count, len(text_array))
        text_tokens = clip.tokenize([text for text in text_array]).cuda()
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).half()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = torch.zeros((1, len(text_array))).to(self.device)
        for i in range(image_features.shape[0]):
            similarity += (100.0 * image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)
        similarity /= image_features.shape[0]

        del image_features

        top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)

        return [(text_array[top_labels[0][i].numpy()], (top_probs[0][i].numpy() * 100)) for i in range(top_count)]

    def get_ranks(self, image_features):
        return [
            self.rank(image_features, self.mediums),
            # self.rank(image_features, ["by "+artist for artist in self.artists]), # memory boom on 10G
            self.rank(image_features, self.trending_list),
            self.rank(image_features, self.movements),
            self.rank(image_features, self.flavors, top_count=3),
        ]

    def generate_caption(self, image):
        gpu_image = (
            transforms.Compose(
                [
                    transforms.Resize((self.blip_image_eval_size, self.blip_image_eval_size), interpolation=InterpolationMode.LANCZOS),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )(image)
            .unsqueeze(0)
            .half()
            .to(self.device)
        )

        with torch.no_grad():
            caption = self.blip_model.generate(gpu_image, sample=False, num_beams=3, max_length=20, min_length=5)
        return caption[0]

    def interrogate(self, image):
        caption = self.generate_caption(image)

        table = []
        bests = [[("", 0)]] * 5

        images = self.preprocess(image).unsqueeze(0).half().cuda()
        with torch.no_grad():
            image_features = self.model.visual(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features.half()

        del images
        self.blip_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        ranks = self.get_ranks(image_features)

        self.blip_model.to(self.device)
        gc.collect()
        torch.cuda.empty_cache()

        for i in range(len(ranks)):
            confidence_sum = 0
            for ci in range(len(ranks[i])):
                confidence_sum += ranks[i][ci][1]
            if confidence_sum > sum(bests[i][t][1] for t in range(len(bests[i]))):
                bests[i] = ranks[i]

        row = [self.model_name]
        for r in ranks:
            row.append(", ".join([f"{x[0]} ({x[1]:0.1f}%)" for x in r]))

        table.append(row)

        print(pd.DataFrame(table, columns=["Model", "Medium", "Trending", "Movement", "Flavors"]))  # removed artists column

        flaves = ", ".join([f"{x[0]}" for x in bests[4]])
        medium = bests[0][0][0]
        if caption.startswith(medium):
            modifiers = f"{bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {flaves}"
        else:
            modifiers = f"{medium} {bests[1][0][0]}, {bests[2][0][0]}, {bests[3][0][0]}, {flaves}"
        print(f"{caption}, {modifiers}")
        return caption, modifiers


def main():
    logging.set_verbosity_error()  # suppress BERT errors

    parser = argparse.ArgumentParser()

    parser.add_argument("-P", "--prompt", type=str, nargs="?", default="a painting of a virus monster playing guitar", help="the prompt to render")
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default="outputs/txt2img-samples")
    parser.add_argument("--skip_grid", action="store_true", help="do not save a grid, only individual samples. Helpful when evaluating lots of samples")
    parser.add_argument("--skip_save", action="store_true", help="do not save individual samples. For speed measurements.")
    parser.add_argument("-N", "--steps", type=int, default=50, help="number of sampling steps")
    parser.add_argument("--kl", action="store_true", help="use k-lms sampling")
    parser.add_argument("--ke", action="store_true", help="use k-euler sampling (default)")
    parser.add_argument("--kea", action="store_true", help="use k-euler-ancestral sampling")
    parser.add_argument("--kh", action="store_true", help="use k-heun sampling")
    parser.add_argument("--kha", action="store_true", help="use k-heun-ancestral sampling")
    parser.add_argument("--kd", action="store_true", help="use k-dpm-2 sampling")
    parser.add_argument("--kda", action="store_true", help="use k-dpm-2-ancestral sampling")
    parser.add_argument("--leaked", action="store_true", help="uses the leaked v1.3 model")
    parser.add_argument("--fixed_code", action="store_true", help="if enabled, uses the same starting code across samples ")
    parser.add_argument("--n-iter", type=int, default=1, help="sample this often")
    parser.add_argument("-H", "--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("-W", "--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--square", action="store_true", help="size preset")
    parser.add_argument("--portrait", action="store_true", help="size preset")
    parser.add_argument("--landscape", action="store_true", help="size preset")
    parser.add_argument("--channels", type=int, default=4, help="latent channels")
    parser.add_argument("--factor", type=int, default=8, help="downsampling factor")
    parser.add_argument("--n-samples", type=int, default=1, help="how many samples to produce for each given prompt. A.k.a. batch size")
    parser.add_argument("--n-rows", type=int, default=0, help="rows in the grid (default: n_samples)")
    parser.add_argument("-C", "--scale", type=float, default=7.5, help="cfg scale - unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")
    parser.add_argument("--from-file", type=str, help="if specified, load prompts from this file")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml", help="path to config which constructs model")
    parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1-4/model.ckpt", help="path to checkpoint of model")
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
    parser.add_argument("--random", action="store_true", help="randomize seed")
    parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
    parser.add_argument("--sigma", type=str, help="sigma scheduler", choices=["old", "karras", "exp", "vp"], default="karras")
    parser.add_argument("--four", action="store_true", help="grid")
    parser.add_argument("--six", action="store_true", help="grid")
    parser.add_argument("--nine", action="store_true", help="grid")
    parser.add_argument("--face", action="store_true", help="gfpgan face fixer")
    parser.add_argument("--skip-normalize", action="store_true", help="normalize prompt weight")
    parser.add_argument("-i", "--init", action="append", help="init images")
    parser.add_argument("--init-factor", type=float, default=0.8, help="strength of the init image, 0.0-1.0")
    parser.add_argument("--interrogate", action="store_true", help="interrogate init images to enhance prompt")
    parser.add_argument("--caption", action="store_true", help="use interrogated caption")
    parser.add_argument("--prefix", action="store_true", help="use interrogated prompt as prefix (default behavior suffix)")
    opt = parser.parse_args()
    assert opt.width % 32 == 0, f"width {opt.width} not a multiple of 32, try {opt.width - (opt.width % 32)}"
    assert opt.height % 32 == 0, f"height {opt.height} not a multiple of 32, try {opt.height - (opt.height % 32)}"
    assert 0.0 <= opt.init_factor <= 1.0, "can only work with init_factor in [0.0, 1.0]"

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

    if opt.random:
        print("Randomized seed!")
        seed_everything(random.getrandbits(32))
    else:
        seed_everything(opt.seed)

    accelerator = accelerate.Accelerator()

    config = OmegaConf.load(f"{opt.config}")

    devicestr = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(devicestr)

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
        print("Sampler set to k-lms")
    elif opt.kea:
        sampler = K.sampling.sample_euler_ancestral
        print("Sampler set to k-euler-ancestral")
    elif opt.kh:
        sampler = K.sampling.sample_heun
        print("Sampler set to k-heun")
    elif opt.kha:
        sampler = K.sampling.sample_heun_ancestral
        print("Sampler set to k-heun-ancestral")
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
        sampler = K.sampling.sample_euler_ancestral
        print("Sampler set to default (k-euler-ancestral)")

    print("{} steps, cfg scale {}, output size {}x{}, {} total iterations".format(opt.steps, opt.scale, opt.width, opt.height, opt.n_iter))

    int_caps = ""
    int_mods = ""
    if opt.init and opt.interrogate:
        inter = Interrogator(devicestr, blip_image_eval_size=512)
        int_caps = []
        int_mods = []
        for i in opt.init:
            print(f"Interrogating {i} ...")
            im = Image.open(i).convert("RGB")
            c, m = inter.interrogate(im)
            m = m.split(",")
            int_caps.append(c)
            int_mods += m
        int_mods = [m.strip() for m in int_mods]
        int_mods = list(dict.fromkeys(int_mods))
        int_mods = ", ".join(int_mods)
        int_caps = list(dict.fromkeys(int_caps))
        int_caps = ", ".join(int_caps)
        del inter
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Interrogated captions: {int_caps}")
        print(f"Interrogated modifiers: {int_mods}")

    GFPGAN = None
    if opt.face:
        if os.path.exists(GFPGAN_dir):
            try:
                GFPGAN = load_GFPGAN()
                GFPGAN.gfpgan.to("cpu")
                GFPGAN.face_helper.face_parse.to("cpu")
                GFPGAN.face_helper.face_det.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
                print("Loaded GFPGAN")
            except Exception:
                import traceback

                print("Error loading GFPGAN:", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)

    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.half()
    model_wrap = K.external.CompVisDenoiser(model)
    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()
    model = model.to(device)

    shape = [opt.channels, opt.height // opt.factor, opt.width // opt.factor]
    init_latent = None
    if opt.init:
        init_latent = torch.zeros([opt.n_samples, *shape], device=device)  # for GPU draw
        for i in opt.init:
            im = load_img(i, opt.width, opt.height).to(device)
            im_latent = model.get_first_stage_encoding(model.encode_first_stage(im))
            init_latent += im_latent / len(opt.init)

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

                        # weighted sub-prompts
                        if opt.interrogate:
                            p = int_mods
                            if opt.caption:
                                p = int_caps + ", " + p
                            p = p.split(",")
                            p = [i.strip() for i in p]
                            p = [i for i in p if i]
                            p = ", ".join(p)
                            if opt.prefix:
                                prompts[0] = p + ", " + prompts[0]
                            else:
                                prompts[0] = prompts[0] + ", " + p
                        print(f"Using prompt: {prompts[0]}")
                        subprompts, weights = split_weighted_subprompts(prompts[0])
                        if len(subprompts) > 1:
                            # i dont know if this is correct.. but it works
                            c = torch.zeros_like(uc)
                            # get total weight for normalizing
                            totalWeight = sum(weights)
                            # normalize each "sub prompt" and add it
                            for i in range(0, len(subprompts)):
                                weight = weights[i]
                                if not opt.skip_normalize:
                                    weight = weight / totalWeight
                                c = torch.add(c, model.get_learned_conditioning(subprompts[i]), alpha=weight)
                        else:  # just standard 1 prompt
                            c = model.get_learned_conditioning(prompts)

                        if opt.sigma == "old":
                            sigmas = model_wrap.get_sigmas(opt.steps)
                        elif opt.sigma == "karras":
                            sigmas = K.sampling.get_sigmas_karras(opt.steps, sigma_min, sigma_max, device=devicestr)
                        elif opt.sigma == "exp":
                            sigmas = K.sampling.get_sigmas_exponential(opt.steps, sigma_min, sigma_max, device=devicestr)
                        elif opt.sigma == "vp":
                            sigmas = K.sampling.get_sigmas_vp(opt.steps, device=devicestr)
                        else:
                            raise ValueError("sigma option error")

                        if opt.init:
                            x = torch.zeros([opt.n_samples, *shape], device=device)  # for GPU draw
                            x += init_latent * opt.init_factor
                            sigmas *= 1 - opt.init_factor
                            x += torch.randn([opt.n_samples, *shape], device=device) * sigmas[0]
                        else:
                            x = torch.randn([opt.n_samples, *shape], device=device) * sigmas[0]  # for GPU draw

                        model_wrap_cfg = CFGDenoiser(model_wrap)
                        extra_args = {"cond": c, "uncond": uc, "cond_scale": opt.scale}
                        samples = sampler(model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=not accelerator.is_main_process)
                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_samp in x_samples:
                                x_samp = 255.0 * rearrange(x_samp.cpu().numpy(), "c h w -> h w c")
                                x_samp = x_samp.astype(np.uint8)

                                if opt.face and GFPGAN is not None:
                                    model.to("cpu")
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    GFPGAN.gfpgan.to("cuda")
                                    GFPGAN.face_helper.face_parse.to("cuda")
                                    GFPGAN.face_helper.face_det.to("cuda")
                                    _, _, restored_img = GFPGAN.enhance(x_samp, has_aligned=False, only_center_face=False, paste_back=True)
                                    x_samp = restored_img
                                    GFPGAN.gfpgan.to("cpu")
                                    GFPGAN.face_helper.face_parse.to("cpu")
                                    GFPGAN.face_helper.face_det.to("cpu")
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    model.to("cuda")
                                    print("Face fixed!")

                                Image.fromarray(x_samp).save(os.path.join(sample_path, f"{base_count:05}.png"))
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_samples)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, "n b c h w -> (n b) c h w")
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f"grid-{grid_count:04}.png"))
                    grid_count += 1

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n" f" \nEnjoy.")


if __name__ == "__main__":
    main()
