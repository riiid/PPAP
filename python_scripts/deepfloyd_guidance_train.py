import argparse



def main():
    return


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
