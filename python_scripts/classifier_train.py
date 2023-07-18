"""
Train a noised image classifier on ImageNet.
"""
import argparse
import copy

import torch.cuda.amp
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    classifier_and_diffusion_defaults,
    args_to_dict,
    create_classifier_and_diffusion,
    add_dict_to_argparser,
    adapter_defaults,
    get_image_normalization,
    experts_defaults
)
from peft.utils import add_adapter_for_classifier


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(method, model, name):
    if method in ["finetune", "multi_experts"]:
        torch.save(model.module.state_dict(), name)
    else:
        torch.save(model.module.state_dict(), name)
        # torch.save(model.module.adapter_state_dict(), name)


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args.gpus)
    logger.configure(dir=args.log_path)

    logger.log("models loading")
    classifier, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )
    classifier.to(dist_util.dev())
    normalize = get_image_normalization(classifier_name=args.classifier_name)

    # Model configuration
    if args.method == "finetune" or args.method == "multi_experts":
        model = DDP(
            classifier,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=False
        )
        model.train()
        model.to(dist_util.dev())
    elif args.method == "pe_multi_experts":
        # add adapter for model
        adapter_classifier = add_adapter_for_classifier(args.classifier_name, classifier,
                                                        gamma=args.gamma, lora_alpha=args.lora_alpha)
        model = DDP(
            adapter_classifier,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=False
        )
        model.train()
    elif args.method == "ppap":
        adapter_classifier = add_adapter_for_classifier(args.classifier_name, copy.deepcopy(classifier),
                                                        gamma=args.gamma, lora_alpha=args.lora_alpha)
        model = DDP(
            adapter_classifier,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=False,
            bucket_cap_mb=128,
            find_unused_parameters=False
        )
        model.train()
        classifier.eval()
    dist_util.sync_params(model.parameters())

    # Optimizers
    logger.log("configuring optimizer")
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logger.log("Data loading")
    # Data configuration
    if args.method in ["finetune", "multi_experts"]:
        dataloader = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=True,
            deterministic=False,
            random_crop=True,
            random_flip=True,
            num_workers=args.num_workers
        )
    elif args.method == "ppap":
        dataloader = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            image_size=args.image_size,
            class_cond=False,
            deterministic=False,
            random_crop=False,
            random_flip=True,
            num_workers=args.num_workers
        )

    # Create diffusion scheduler for forward diffusion.
    logger.log("Create diffusion scheduler")
    schedule_sampler = create_named_schedule_sampler(
        args.schedule_sampler, diffusion
    )
    if args.method == "finetune":
        args.n_experts = 1

    args.iterations = int(args.iterations / args.n_experts)
    logger.log(f"train each expert on {args.iterations}")

    range_list = []
    max_step = diffusion.num_timesteps
    for interval in range(args.n_experts):
        now_range = [
            int(interval / args.n_experts * max_step), int((interval + 1) / args.n_experts * max_step)]
        range_list.append(now_range)

    # Do training-loop
    logger.log("training!")
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
    for interval in range(args.n_experts):
        # Set target timestep range for each experts.
        now_range = range_list[interval]
        schedule_sampler._weights[:] = 0
        schedule_sampler._weights[now_range[0]: now_range[1]] = 1

        # batch set-up
        for step in tqdm(range(int(args.iterations))):
            if args.anneal_lr:
                lr = set_annealed_lr(opt, base_lr=args.lr, frac_done=step / int(args.iterations))
            else:
                lr = args.lr

            logger.logkv("step", step)
            logger.logkv("samples", (step + 1) * args.batch_size * dist.get_world_size())
            logger.logkv("lr", lr)

            opt.zero_grad(set_to_none=True)
            try:
                batch = next(batch_iterator)
            except:
                batch_iterator = iter(dataloader)
                batch = next(batch_iterator)

            if args.method in ["finetune", "multi_experts"]:
                data, extra = batch
                data = data.to(dist_util.dev())
                labels = extra["y"].to(dist_util.dev())

                # forward process
                t, _ = schedule_sampler.sample(data.shape[0], dist_util.dev())
                data = diffusion.q_sample(data, t)

                with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    logits = model(data)
                    loss = F.cross_entropy(logits, labels, reduction="none")

                scaler.scale(loss.mean()).backward()
                scaler.step(opt)
            else:
                data, extra = batch
                data = data.to(dist_util.dev())
                with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    with torch.no_grad():
                        data_for_origin_classifier = normalize((data + 1) / 2)
                        output_origin = classifier(data_for_origin_classifier)
                        t, _ = schedule_sampler.sample(data_for_origin_classifier.shape[0], dist_util.dev())
                        forward_img = diffusion.q_sample(data, t)
                    output = model(forward_img)
                    KD_loss = torch.nn.KLDivLoss(reduction="none")(
                        F.log_softmax(output, dim=1),
                        F.softmax(output_origin / 0.25, dim=1)
                    )
                    loss = KD_loss.mean(dim=-1)
                scaler.scale(loss.mean()).backward()
                scaler.step(opt)
            scaler.update()
            losses = {}
            losses[f"loss"] = loss.detach()
            log_loss_dict(diffusion, t, losses)
            logger.logkv("scale", scaler.get_scale())
            if not step % args.log_interval:
                logger.dumpkvs()
            if (
                dist.get_rank() == 0
                and not (step) % args.save_interval
            ):
                logger.log("saving model during training")
                save_checkpoint(args.method, model, logger.get_dir() + f"/{interval}_iteration{step}_max{args.n_experts}.ckpt")

        if dist.get_rank() == 0:
            logger.log("saving final expert")
            save_checkpoint(args.method, model, logger.get_dir() + f"/{interval}_max{args.n_experts}.ckpt")
        dist.barrier()


def create_argparser():
    defaults = dict(
        image_size=256,
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        save_interval=10000,
        method="finetune",  # 1) finetune, 2) multi_experts, 3) ppap
        log_path="logs",
        mixed_precision=True,
        num_workers=16
    )
    defaults.update(classifier_and_diffusion_defaults())
    defaults.update(adapter_defaults())
    defaults.update(experts_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, nargs="+", default="0")
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
