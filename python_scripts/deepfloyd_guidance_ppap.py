import argparse
from copy import deepcopy
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from deepfloyd_if.script_utils import create_model_and_diffusion, create_guidance_model, add_adapter_for_guidance_model
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import add_dict_to_argparser


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args.gpus)
    logger.configure(dir=args.log_path)

    logger.log("create diffusion")
    _, diffusion = create_model_and_diffusion(
        token=args.token
    )

    logger.log("create model")
    original_model = create_guidance_model(task_name=args.external_model)
    adapter_model = add_adapter_for_guidance_model(task_name=args.external_model,
                                                   model=deepcopy(original_model),
                                                   lora_alpha=args.lora_alpha,
                                                   gamma=args.lora_gamma
                                   )

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
        log_path="logs"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, nargs="+", default="0")
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__=="__main__":
    main()
