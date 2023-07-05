"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.

We start to modify codes from (https://github.com/openai/guided-diffusion/blob/main/scripts/classifier_sample.py).
"""
import os
import argparse
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import copy

from guided_diffusion import dist_util, logger
from guided_diffusion.multi_expert_helper import MultiExpertWrapper
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    args_to_dict,
    create_pretrained_classifier,
    adapter_defaults,
    experts_defaults
)
from peft.utils import add_adapter_for_classifier


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

    logger.log("loading classifier...")
    classifier = create_pretrained_classifier(args.classifier_name)

    if args.method in ["ppap"]:
        classifier = add_adapter_for_classifier(args.classifier_name, classifier,
                                                    gamma=args.gamma, lora_alpha=args.lora_alpha)
    if len(args.classifier_path) > 1:
        models = []
        for ckpt_path in args.classifier_path:
            logger.log(f"load {ckpt_path}")
            m, u = classifier.load_state_dict(torch.load(ckpt_path), strict=False)
            print("missing keys", m)
            print("unexpected keys",u)
            classifier.eval()
            classifier.to(dist_util.dev())
            models.append(copy.deepcopy(classifier))
        classifier = MultiExpertWrapper(models, len(args.classifier_path), diffusion.original_num_steps)
    elif len(args.classifier_path) == 1:
        classifier.load_state_dict(torch.load(args.classifier_path[0]), strict=False)
        classifier.to(dist_util.dev())
        classifier.eval()
    else:
        classifier.to(dist_util.dev())
        classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            if len(args.classifier_path) > 1:
                logits = classifier(x_in, t)
            else:
                logits = classifier(x_in)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            grad = torch.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale
            return grad

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)


    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = torch.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        with torch.no_grad():
            sample = sample_fn(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=dist_util.dev(),
            )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        gathered_labels = [torch.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)

        if dist.get_rank() == 0:
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        else:
            all_labels.extend([1 for _ in gathered_labels])
            all_images.extend([1 for _ in gathered_samples])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    if dist.get_rank() == 0:
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)
    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_scale=7.5,
        log_path="sample_logs",
        method="finetune"   # 1) finetune, 2) multi_experts 3) ppap
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    defaults.update(adapter_defaults())
    defaults.update(experts_defaults())
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--classifier_path", type=str, nargs="+")
    parser.add_argument("--gpus", type=str, nargs="+", default="0")
    return parser


if __name__ == "__main__":
    main()
