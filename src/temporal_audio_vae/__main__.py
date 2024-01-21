import logging
import argparse
from pathlib import Path
from temporal_audio_vae.helpers.colorlogger import Colorformatter
import torch

# setup logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(Colorformatter())
logger.addHandler(ch)
logging.getLogger("numba").setLevel(logging.INFO)


def do_train(args):
    if args.beta and args.warmup:
        logger.critical("Cannot use fixed beta and beta warmup at the same time")
        exit(1)

    if not (args.dbpath / "data.mdb").exists():
        logger.critical(f"Could not find 'data.mdb' in '{args.dbpath}'")
        exit(1)

    if args.checkpoint is None and args.beta is None and args.warmup is None:
        logger.critical(
            "Must at least use fixed beta or beta warmup, or use a checkpoint"
        )
        exit(1)

    from .train import Trainer

    trainer = Trainer(
        dataset_path=args.dbpath,
        beta=args.beta,
        use_beta_warmup=args.warmup is not None,
        warmup_epoch_interval=(args.warmup[0], args.warmup[1]) if args.warmup else None,
        warmup_beta_interval=(args.warmup[2], args.warmup[3]) if args.warmup else None,
    )
    if args.checkpoint:
        trainer.load_state(args.checkpoint)

    trainer.train(
        epoch_end=args.endepoch,
        evaluate_every=args.validepoch,
        generate_every=args.genepoch,
        save_every=args.saveepoch,
    )


def do_gendataset(args):
    if not (args.dbpath / "data.mdb").exists():
        logger.critical(f"Could not find 'data.mdb' in '{args.dbpath}'")
        exit(1)
    from .train import Trainer
    import torchaudio
    import torchvision.utils

    trainer = Trainer(dataset_path=args.dbpath)
    trainer.load_state(args.checkpoint)
    waveform = torch.stack([trainer.dataset[args.id]])
    mag, _ = trainer.transform(waveform)
    grid = torchvision.utils.make_grid(
        mag.reshape(-1, 1, trainer.n_mels, trainer.n_frames),
        waveform.shape[0],
        pad_value=1,
    )
    torchvision.utils.save_image(grid, f"original_melspec_{args.id}.png")
    torchaudio.save(f"original_{args.id}.wav", waveform, trainer.dataset.FS)
    
    copyphase, griffinlim, image = trainer.generate_from_dataset(waveform)
    torchaudio.save(f"copyphase_{args.id}.wav", copyphase, trainer.dataset.FS)
    torchaudio.save(f"griffinlim_{args.id}.wav", griffinlim, trainer.dataset.FS)
    torchvision.utils.save_image(image, f"reconstructed_melspec_{args.id}.png")


def do_genrandom(args):
    if not (args.dbpath / "data.mdb").exists():
        logger.critical(f"Could not find 'data.mdb' in '{args.dbpath}'")
        exit(1)
    from .train import Trainer
    import torchaudio
    import torchvision.utils

    trainer = Trainer(dataset_path=args.dbpath)
    trainer.load_state(args.checkpoint)
    waveform, image = trainer.generate_latent_random(args.n)
    torchaudio.save(f"random.wav", waveform, trainer.dataset.FS)
    torchvision.utils.save_image(image, f"random.png")


def do_genlatent(args):
    if not (args.dbpath / "data.mdb").exists():
        logger.critical(f"Could not find 'data.mdb' in '{args.dbpath}'")
        exit(1)
    from .train import Trainer
    import torchaudio
    import torchvision.utils

    trainer = Trainer(dataset_path=args.dbpath)
    trainer.load_state(args.checkpoint)
    waveform, image = trainer.generate_latent_explore(args.n_per_dimension)
    torchaudio.save(f"explore.wav", waveform, trainer.dataset.FS)
    torchvision.utils.save_image(image, f"explore.png")


parser = argparse.ArgumentParser()
parser.add_argument("--dbpath", type=Path, required=True)
parser.add_argument(
    "--checkpoint", type=Path, help="continue from checkpoint", metavar="PATH"
)
subparsers = parser.add_subparsers(required=True)

## TRAIN
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
    "--validepoch", type=int, help="validate every nth epoch", metavar="N", default=10
)
train_parser.add_argument(
    "--saveepoch", type=int, help="save every nth epoch", metavar="N", default=10
)

## GENERATE
gendataset_parser = subparsers.add_parser("gendataset")
gendataset_parser.set_defaults(func=do_gendataset)
gendataset_parser.add_argument("id", type=int)

gendataset_parser = subparsers.add_parser("genrandom")
gendataset_parser.set_defaults(func=do_genrandom)
gendataset_parser.add_argument("n", type=int)

gendataset_parser = subparsers.add_parser("genlatent")
gendataset_parser.set_defaults(func=do_genlatent)
gendataset_parser.add_argument("n_per_dimension", type=int)

args = parser.parse_args()


args.func(args)
