"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.

We start to modify codes from (https://github.com/openai/guided-diffusion/blob/main/scripts/classifier_sample.py).
"""
import os
import argparse
import torch
import torch.distributed as dist
from PIL import Image

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser
)


NUM_CLASSES=1000
def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args.gpus)
    logger.configure(dir=args.log_path)
    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    def model_fn(x, t, y=None):
        return model(x, t, y if args.class_cond else None)


    logger.log("sampling...")
    n_saved = 0
    while n_saved * dist.get_world_size() + dist.get_rank() < args.num_samples:
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        with torch.no_grad():
            sample = sample_fn(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                device=dist_util.dev(),
            )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        for im in sample:
            imgpath = os.path.join(logger.get_dir(), f"{n_saved * dist.get_world_size() + dist.get_rank():06}.png")
            if n_saved * dist.get_world_size() + dist.get_rank() > args.num_samples:
                break
            i = Image.fromarray(im.cpu().numpy()).convert("RGB")
            i.save(imgpath)
            n_saved += 1
        logger.log(f"{n_saved * dist.get_world_size() + dist.get_rank()} images are saved.")
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=500000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        log_path="sample_logs",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--gpus", type=str, nargs="+", default="0")
    return parser


if __name__ == "__main__":
    main()
