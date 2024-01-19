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

    train(args.dbpath)


parser = argparse.ArgumentParser()
parser.add_argument("--dbpath", type=Path, required=True)
subparsers = parser.add_subparsers(required=True)

train_parser = subparsers.add_parser("train")
train_parser.set_defaults(func=do_train)

args = parser.parse_args()

if not (args.dbpath / "data.mdb").exists():
    logger.critical(f"Could not find 'data.mdb' in '{args.dbpath}'")
    exit(1)

args.func(args)
