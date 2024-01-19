import logging
import argparse
from pathlib import Path
from temporal_audio_vae.helpers.colorlogger import Colorformatter

# setup logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(Colorformatter())
logger.addHandler(ch)


def do_train(args):
    logging.info("Training")
    from .train import train

    train(
        dataset_path=args.dbpath,
        beta=args.beta,
        use_beta_warmup=args.warmup is not None,
        warmup_epoch_interval=(args.warmup[0], args.warmup[1]) if args.warmup else None,
        warmup_beta_interval=(args.warmup[2], args.warmup[3]) if args.warmup else None,
        epoch_start=1,
        epoch_end=args.endepoch,
    )


parser = argparse.ArgumentParser()
parser.add_argument("--dbpath", type=Path, required=True)
subparsers = parser.add_subparsers(required=True)

train_parser = subparsers.add_parser("train")
train_parser.set_defaults(func=do_train)
train_parser.add_argument("--beta", type=float, help="use fixed beta value")
train_parser.add_argument(
    "--warmup",
    type=float,
    help="use linear beta warmup from EPOCH_0 (beta=BETA_0) to EPOCH_1 (beta=BETA_1)",
    metavar=("EPOCH_0", "EPOCH_1", "BETA_0", "BETA_1"),
    nargs=4,
)
train_parser.add_argument(
    "--endepoch", type=int, help="Stop at speicified epoch", metavar="N"
)
train_parser.add_argument(
    "--genepoch", type=int, help="generate every nth epoch", metavar="N"
)
train_parser.add_argument(
    "--validepoch", type=int, help="validate every nth epoch", metavar="N"
)

args = parser.parse_args()

if args.beta and args.warmup:
    logger.critical("Cannot use fixed beta and beta warmup at the same time")
    exit(1)

if args.beta is None and args.warmup is None:
    logger.critical("Must at least use fixed beta or beta warmup")
    exit(1)

if not (args.dbpath / "data.mdb").exists():
    logger.critical(f"Could not find 'data.mdb' in '{args.dbpath}'")
    exit(1)

args.func(args)
