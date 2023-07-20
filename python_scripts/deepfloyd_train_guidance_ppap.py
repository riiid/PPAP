import argparse
from copy import deepcopy

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm import tqdm
import torch.distributed as dist

from deepfloyd_if.ppap_data.ppap_data import load_data
from deepfloyd_if.script_utils import create_model_and_diffusion, create_guidance_model, add_adapter_for_guidance_model, \
    get_image_normalization
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import add_dict_to_argparser, adapter_defaults, experts_defaults
from python_scripts.classifier_train import set_annealed_lr, log_loss_dict


def save_checkpoint(model, name):
    torch.save(model.module.state_dict(), name)


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args.gpus)
    logger.configure(dir=args.log_path)

    logger.log("create diffusion")
    _, diffusion = create_model_and_diffusion(token=args.token)

    logger.log("create model")
    original_model = create_guidance_model(task_name=args.external_model)
    adapter_model = add_adapter_for_guidance_model(task_name=args.external_model,
                                                   model=deepcopy(original_model),
                                                   lora_alpha=args.lora_alpha,
                                                   gamma=args.gamma
                                   )
    normalize = get_image_normalization(task_name=args.external_model)
    adapter_model.to(dist_util.dev())
    original_model.to(dist_util.dev())
    model = DDP(
        adapter_model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False
    )
    model.train()
    original_model.eval()
    dist_util.sync_params(model.parameters())

    logger.log("configuring optimizer")
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logger.log("loading data")
    dataloader, sampler = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    args.iterations = int(args.iterations / args.n_experts)
    logger.log(f"train each expert on {args.iterations}")

    range_list = []
    max_step = diffusion.num_timesteps
    for interval in range(args.n_experts):
        now_range = [
            int(interval / args.n_experts * max_step), int((interval + 1) / args.n_experts * max_step)]
        range_list.append(now_range)

    logger.log("create scheduler")
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    logger.log("training!")
    epoch = 0
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
                if sampler:
                    sampler.set_epoch(epoch)
                    epoch += 1
                batch_iterator = iter(dataloader)
                batch = next(batch_iterator)

            # Loss function
            low_res_img = batch["I"].to(dist_util.dev())
            high_res_img = batch["II"].to(dist_util.dev())

            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                with torch.no_grad():
                    data_for_origin_model = normalize((high_res_img + 1) / 2)
                    origin_output = original_model(data_for_origin_model)
                t, _ = schedule_sampler.sample(low_res_img.shape[0], dist_util.dev())
                forward_image = diffusion.q_sample(low_res_img, t)
                output = model(forward_image)
                loss = torch.nn.MSELoss(reduction="none")(
                    output, origin_output
                ).mean(dim=-1).mean(dim=-1)
            scaler.scale(loss.mean()).backward()
            scaler.step(opt)
            scaler.update()
            # logging
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
                save_checkpoint(model, logger.get_dir() + f"/{interval}_iteration{step}_max{args.n_experts}.ckpt")
        if dist.get_rank():
            logger.log("saving final expert")
            save_checkpoint(model, logger.get_dir() + f"/{interval}_max{args.n_experts}.ckpt")
        dist.barrier()


def create_argparser():
    defaults = dict(
        data_dir="/gen_samples_IF",
        lr=1e-4,
        model_path="",
        weight_decay=0.05,
        batch_size=1,
        log_interval=50,
        save_interval=10000,
        iterations=300000,
        external_model="depth", # depth,
        token=None,
        log_path="logs",
        num_workers=16,
        mixed_precision=True,
        anneal_lr=True
    )
    defaults.update(adapter_defaults())
    defaults.update(experts_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, nargs="+", default="0")
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__=="__main__":
    main()
