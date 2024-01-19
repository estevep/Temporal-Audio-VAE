from typing import Tuple
from .datasets import LoopDataset
from .models import MelSpecVAE, construct_encoder_decoder
from .transforms import Log1pMelSpecPghi
from .helpers import beta_warmup, find_normalizer
from .generate_rand import generate_rand
from .generate_data import generate_data
from .generate_explore import generate_explore
import torch
import logging
import torchvision
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def train(
    dataset_path: str,
    beta: float = 0,
    use_beta_warmup: bool = False,
    warmup_epoch_interval: Tuple[float, float] = None,
    warmup_beta_interval: Tuple[float, float] = None,
    epoch_start: int = 1,
    epoch_end: int = None,
    evaluate_every_nth_epoch: int = None,
    generate_every_nth_epoch: int = None,
    n_sounds_per_dimension: int = None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: %s", device)

    # load dataset
    dataset = LoopDataset(dataset_path)
    train_loader, valid_loader = dataset.get_loaders()

    # hyperparameters
    n_latent = 16
    n_hidden = 128
    n_mels = 512
    n_fft = 1024
    griffin_lim_iter = 64
    hop_length = 256
    n_sounds_generated_from_dataset = 4
    n_sounds_generated_from_random = 8

    # transform
    transform = Log1pMelSpecPghi(
        sample_rate=LoopDataset.FS,
        n_mels=n_mels,
        n_fft=n_fft,
        griffin_lim_iter=griffin_lim_iter,
        hop_length=hop_length,
    )

    n_frames = transform.get_n_frames(LoopDataset.LEN_SAMPLES)

    # normalize dataset
    train_norm = find_normalizer(train_loader, "train", transform).to(device)
    valid_norm = find_normalizer(valid_loader, "valid", transform).to(device)

    # Build the VAE model
    encoder, decoder = construct_encoder_decoder(n_hidden=n_hidden, n_latent=n_latent)
    model = MelSpecVAE(encoder, decoder, n_hidden, n_latent).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    recons_criterion = torch.nn.MSELoss(reduction="sum")
    transform = transform.to(device)

    WRITER = SummaryWriter(comment="train")

    # add refs to tensorboard
    with torch.no_grad():
        waveform = next(iter(valid_loader)).to(device)
        mag, phase = transform.forward(waveform[:n_sounds_generated_from_dataset])
        mag = train_norm(mag)

        grid = torchvision.utils.make_grid(mag.reshape(-1, 1, n_mels, n_frames), 1)
        WRITER.add_image("ref/dataset/melspec", grid)

        WRITER.add_audio(
            "ref/dataset/original",
            waveform[:n_sounds_generated_from_dataset].reshape(-1),
            0,
            sample_rate=LoopDataset.FS,
        )

    logger.info("BEGINNING TRAINING")
    if use_beta_warmup:
        logger.info(
            f"Using beta warmup: {warmup_epoch_interval=}, {warmup_beta_interval=}"
        )
    else:
        logger.info(f"Using fixed {beta=:.2f}")
    logger.info(f"{epoch_start=} {epoch_end=}")
    logger.info(f"{evaluate_every_nth_epoch=}")
    logger.info(f"{generate_every_nth_epoch=}")
    logger.info(f"{n_latent=}")
    logger.info(f"{n_hidden=}")
    logger.info(f"{n_mels=}")
    logger.info(f"{n_fft=}")
    logger.info(f"{griffin_lim_iter=}")
    logger.info(f"{hop_length=}")
    logger.info(f"{n_sounds_generated_from_dataset=}")
    logger.info(f"{n_sounds_generated_from_random=}")

    ### TRAINING LOOP
    epoch = epoch_start
    while epoch_end is None or epoch < epoch_end:
        model.train()

        if use_beta_warmup:
            beta = beta_warmup(epoch, warmup_beta_interval, warmup_epoch_interval)

        logger.info(f"Training epoch = {epoch}, beta={beta:.2f}")
        full_loss = 0
        recons_loss = 0
        kl_div = 0

        for i, waveform in enumerate(train_loader):
            waveform = waveform.to(device)
            mag, phase = transform.forward(waveform)
            mag = train_norm(mag)

            mag_tilde, kl_div_batch = model(mag)

            recons_loss_batch = recons_criterion(mag_tilde, mag)
            full_loss_batch = recons_loss_batch - beta * kl_div_batch

            recons_loss += recons_loss_batch
            full_loss += full_loss_batch
            kl_div += kl_div_batch

            optimizer.zero_grad()
            full_loss_batch.backward()
            optimizer.step()
        WRITER.add_scalar("loss/train/full", full_loss, epoch)
        WRITER.add_scalar("loss/train/reconstruction", recons_loss, epoch)
        WRITER.add_scalar("loss/train/kl_div", kl_div, epoch)

        ## EVALUATION
        if evaluate_every_nth_epoch and epoch % evaluate_every_nth_epoch == 0:
            model.eval()
            logger.info("Evaluating model")

            full_loss = 0
            recons_loss = 0
            kl_div = 0
            for i, waveform in enumerate(valid_loader):
                waveform = waveform.to(device)
                mag, phase = transform.forward(waveform)
                mag = valid_norm(mag)

                mag_tilde, kl_div_batch = model(mag)

                recons_loss_batch = recons_criterion(mag_tilde, mag)
                full_loss_batch = recons_loss_batch - beta * kl_div_batch

                recons_loss += recons_loss_batch
                full_loss += full_loss_batch
                kl_div += kl_div_batch

                optimizer.zero_grad()
                full_loss_batch.backward()
                optimizer.step()
            WRITER.add_scalar("loss/valid/full", full_loss, epoch)
            WRITER.add_scalar("loss/valid/reconstruction", recons_loss, epoch)
            WRITER.add_scalar("loss/valid/kl_div", kl_div, epoch)

        if generate_every_nth_epoch and epoch % generate_every_nth_epoch == 0:
            
            logger.info("generating from dataset")
            waveform_tilde_copyphase, waveform_tilde_griffinlim, grid = generate_data(model=model, 
                                                                                        transform=transform,
                                                                                        valid_loader=valid_loader,
                                                                                        n_sounds_generated_from_dataset=n_sounds_generated_from_dataset)
            
            WRITER.add_image("gen/dataset/melspec", grid, epoch)
            WRITER.add_audio(
                "gen/dataset/copyphase",
                waveform_tilde_copyphase.reshape(-1),
                epoch,
                sample_rate=LoopDataset.FS,
            )
            WRITER.add_audio(
                    "gen/dataset/griffinlim",
                    waveform_tilde_griffinlim.reshape(-1),
                    epoch,
                    sample_rate=LoopDataset.FS,
                )

            
            logger.info("generating random from latent space")
            waveform_tilde_griffinlim, grid = generate_rand(model=model, 
                                                            transform=transform, 
                                                            n_sounds_generated_from_random=n_sounds_generated_from_random, 
                                                            n_latent=n_latent
            )

            WRITER.add_image("gen/rand/melspec", grid, epoch)
            WRITER.add_audio(
                "gen/rand/copyphase",
                waveform_tilde_copyphase.reshape(-1),
                epoch,
                sample_rate=LoopDataset.FS,
            )
            
            logger.info("exploring latent space")
           
            waveform_tilde_griffinlim, grid = generate_explore(model=model,
                                                              transform=transform,
                                                              n_sounds_per_dimension=3,
                                                              n_latent=n_latent
                                                              )
            
            WRITER.add_image("gen/explo_latent/melspec", grid, epoch)
            WRITER.add_audio(
                "gen/explo_latent/griffinlim",
                waveform_tilde_griffinlim.reshape(-1),
                epoch,
                sample_rate=LoopDataset.FS,
            )
        
        epoch += 1
    ### END TRAINING LOOP
