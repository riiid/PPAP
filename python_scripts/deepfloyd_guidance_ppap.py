import argparse

from guided_diffusion import dist_util, logger


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args.gpus)
    logger.configure(dir=args.log_path)

    logger.log("create diffusion")



def create_argparser():
    defaults = dict(
        data_dir="/gen_samples_IF",
        lr=1e-4,
        model_path="",
        weight_decay=0.05,
        batch_size=1,
        log_interval=50,
        save_interval=10000,
        iterations=300000
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_offset", default=0)
    return parser


if __name__=="__main__":
    main()
