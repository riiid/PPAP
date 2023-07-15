import argparse
import os

import torch
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import add_dict_to_argparser


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args.gpus)
    logger.configure(dir=args.log_path)

    from deepfloyd_if.modules import IFStageI, IFStageII, StableStageIII
    from deepfloyd_if.pipelines.uncond_dream import uncond_dream
    if args.hf_token:
        from huggingface_hub import login
        login(token=args.hf_token)
    logger.log("creating diffusion models...")
    logger.log(f"stage: {args.stage}")
    if_I = None
    if_II = None
    if args.stage >= 1:
        os.makedirs(os.path.join(logger.get_dir(), "I"), exist_ok=True)
        if_I = IFStageI('IF-I-L-v1.0', device=dist_util.dev())
    if args.stage >= 2:
        os.makedirs(os.path.join(logger.get_dir(), "II"), exist_ok=True)
        if_II = IFStageII('IF-II-M-v1.0', device=dist_util.dev())

    n_saved = 0
    while n_saved * dist.get_world_size() + dist.get_rank() < args.num_samples:
        result = uncond_dream(
            if_I=if_I, if_II=if_II, if_III=None,
            batch_size=args.batch_size,
            if_I_kwargs={
                "sample_timestep_respacing": "smart100",
                "sample_loop": "ddim"
            },
            if_II_kwargs={
                "sample_timestep_respacing": "smart50",
                "sample_loop": "ddim"
            },
            disable_watermark=True,
            return_tensors=False
        )
        img = result

        for key, value in img.items():
            n_saved_tmp = n_saved
            for i in value:
                imgpath = os.path.join(logger.get_dir(), f"{key}/{n_saved_tmp * dist.get_world_size() + dist.get_rank():06}.png")
                if n_saved_tmp * dist.get_world_size() + dist.get_rank() > args.num_samples:
                    break
                i.save(imgpath)
                n_saved_tmp += 1
        n_saved = n_saved_tmp
        logger.log(f"{n_saved * dist.get_world_size() + dist.get_rank()} images are saved.")
    logger.log("sampling complete")


def create_argparser():
    default = dict(
        num_samples=10000,
        batch_size=10,
        log_path="sample_logs",
        hf_token=None,
        stage=3
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default)
    parser.add_argument("--gpus", type=str, nargs="+", default="0")
    return parser


if __name__ == "__main__":
    main()
