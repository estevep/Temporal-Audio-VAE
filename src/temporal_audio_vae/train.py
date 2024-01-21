from pathlib import Path
from typing import Tuple
from .datasets import LoopDataset
from .models import MelSpecVAE, construct_encoder_decoder
from .transforms import Log1pMelSpecPghi
from .helpers import beta_warmup, find_normalizer
import torch
import logging
import torchvision
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        dataset_path: str,
        beta: float = 0,
        use_beta_warmup: bool = False,
        warmup_epoch_interval: Tuple[float, float] = None,
        warmup_beta_interval: Tuple[float, float] = None,
    ) -> None:
        self.beta = beta
        self.use_beta_warmup = use_beta_warmup
        self.warmup_epoch_interval = warmup_epoch_interval
        self.warmup_beta_interval = warmup_beta_interval

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", self.device)

        # load dataset
        self.dataset = LoopDataset(dataset_path)
        self.train_loader, self.valid_loader = self.dataset.get_loaders()

        # hyperparameters
        self.n_latent = 16
        self.n_hidden = 128
        self.n_mels = 512
        self.n_fft = 1024
        self.griffin_lim_iter = 64
        self.hop_length = 256
        self.n_sounds_generated_from_dataset = 4
        self.n_sounds_generated_from_random = 8

        # transform
        self.transform = Log1pMelSpecPghi(
            sample_rate=LoopDataset.FS,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            griffin_lim_iter=self.griffin_lim_iter,
            hop_length=self.hop_length,
        )

        self.n_frames = self.transform.get_n_frames(LoopDataset.LEN_SAMPLES)
        self.epoch = 1

        # normalize dataset
        self.train_norm = find_normalizer(
            self.train_loader, "train", self.transform
        ).to(self.device)
        self.valid_norm = find_normalizer(
            self.valid_loader, "valid", self.transform
        ).to(self.device)

        # Build the VAE model
        encoder, decoder = construct_encoder_decoder(
            n_hidden=self.n_hidden, n_latent=self.n_latent
        )
        self.model = MelSpecVAE(encoder, decoder, self.n_hidden, self.n_latent).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.recons_criterion = torch.nn.MSELoss(reduction="sum")
        self.transform = self.transform.to(self.device)

        self.writer = SummaryWriter(comment="train")

    def dump(self):
        if self.use_beta_warmup:
            logger.info(
                f"Using beta warmup: {self.warmup_epoch_interval=}, {self.warmup_beta_interval=}"
            )
        else:
            logger.info(f"Using fixed {self.beta=:.2f}")
        logger.info(f"{self.epoch=}")
        logger.info(f"{self.n_latent=}")
        logger.info(f"{self.n_hidden=}")
        logger.info(f"{self.n_mels=}")
        logger.info(f"{self.n_fft=}")
        logger.info(f"{self.griffin_lim_iter=}")
        logger.info(f"{self.hop_length=}")
        logger.info(f"{self.n_sounds_generated_from_dataset=}")
        logger.info(f"{self.n_sounds_generated_from_random=}")

    def train(
        self,
        epoch_end: int = None,
        evaluate_every: int = None,
        generate_every: int = None,
        save_every: int = None,
    ):
        # add refs to tensorboard
        with torch.no_grad():
            waveform = next(iter(self.train_loader)).to(self.device)
            mag, phase = self.transform.forward(
                waveform[: self.n_sounds_generated_from_dataset]
            )
            mag = self.train_norm(mag)

            grid = torchvision.utils.make_grid(
                mag.reshape(-1, 1, self.n_mels, self.n_frames),
                self.n_sounds_generated_from_dataset,
                pad_value=1,
            )
            self.writer.add_image("ref/dataset/melspec", grid)

            self.writer.add_audio(
                "ref/dataset/original",
                waveform[: self.n_sounds_generated_from_dataset].reshape(-1),
                0,
                sample_rate=LoopDataset.FS,
            )

        logger.info("BEGINNING TRAINING")

        ### TRAINING LOOP
        while epoch_end is None or self.epoch < epoch_end:
            self.model.train()

            if self.use_beta_warmup:
                self.beta = beta_warmup(
                    self.epoch, self.warmup_beta_interval, self.warmup_epoch_interval
                )

            logger.info(f"Training epoch = {self.epoch}, beta={self.beta:.2f}")
            full_loss = 0
            recons_loss = 0
            kl_div = 0

            for waveform in self.train_loader:
                waveform = waveform.to(self.device)
                mag, _ = self.transform.forward(waveform)
                mag = self.train_norm(mag)

                mag_tilde, kl_div_batch = self.model(mag)

                recons_loss_batch = self.recons_criterion(mag_tilde, mag)
                full_loss_batch = recons_loss_batch - self.beta * kl_div_batch

                recons_loss += recons_loss_batch
                full_loss += full_loss_batch
                kl_div += kl_div_batch

                self.optimizer.zero_grad()
                full_loss_batch.backward()
                self.optimizer.step()
            self.writer.add_scalar("loss/train/full", full_loss, self.epoch)
            self.writer.add_scalar("loss/train/reconstruction", recons_loss, self.epoch)
            self.writer.add_scalar("loss/train/kl_div", kl_div, self.epoch)

            ## EVALUATION
            if evaluate_every and self.epoch % evaluate_every == 0:
                self.model.eval()
                logger.info("Evaluating model")

                full_loss = 0
                recons_loss = 0
                kl_div = 0
                for waveform in self.valid_loader:
                    waveform = waveform.to(self.device)
                    mag, _ = self.transform.forward(waveform)
                    mag = self.valid_norm(mag)

                    mag_tilde, kl_div_batch = self.model(mag)

                    recons_loss_batch = self.recons_criterion(mag_tilde, mag)
                    full_loss_batch = recons_loss_batch - self.beta * kl_div_batch

                    recons_loss += recons_loss_batch
                    full_loss += full_loss_batch
                    kl_div += kl_div_batch

                    self.optimizer.zero_grad()
                    full_loss_batch.backward()
                    self.optimizer.step()
                self.writer.add_scalar("loss/valid/full", full_loss, self.epoch)
                self.writer.add_scalar(
                    "loss/valid/reconstruction", recons_loss, self.epoch
                )
                self.writer.add_scalar("loss/valid/kl_div", kl_div, self.epoch)

            if save_every and self.epoch % save_every == 0:
                self.save()

            #### GENERATION
            if generate_every and self.epoch % generate_every == 0:
                self.generate_from_dataset(
                    next(iter(self.train_loader)).to(self.device)
                )
                self.generate_latent_random()
                self.generate_latent_explore()

            self.epoch += 1

    def generate_from_dataset(self, waveform):
        logger.info("generating from dataset")
        with torch.no_grad():
            train_norm = find_normalizer(self.train_loader, "train", self.transform).to(
                self.device
            )

            mag, phase = self.transform.forward(waveform)
            mag = train_norm(mag)

            mag_tilde, _ = self.model(mag)

            waveform_tilde_copyphase = self.transform.backward(mag_tilde, phase)
            waveform_tilde_griffinlim = self.transform.backward(mag_tilde)

            grid = torchvision.utils.make_grid(
                mag_tilde.reshape(-1, 1, self.n_mels, self.n_frames),
                waveform.shape[0],
                pad_value=1,
            )

            waveform_tilde_copyphase /= torch.max(abs(waveform_tilde_copyphase))
            waveform_tilde_griffinlim /= torch.max(abs(waveform_tilde_griffinlim))

        self.writer.add_image("gen/dataset/melspec", grid, self.epoch)
        self.writer.add_audio(
            "gen/dataset/copyphase",
            waveform_tilde_copyphase.reshape(-1),
            self.epoch,
            sample_rate=LoopDataset.FS,
        )
        self.writer.add_audio(
            "gen/dataset/griffinlim",
            waveform_tilde_griffinlim.reshape(-1),
            self.epoch,
            sample_rate=LoopDataset.FS,
        )
        return waveform_tilde_copyphase, waveform_tilde_griffinlim, grid

    def generate_latent_random(self, n_sounds_generated_from_random):
        logger.info("generating random from latent space")

        with torch.no_grad():
            z = torch.randn(n_sounds_generated_from_random, self.n_latent).to(
                self.device
            )
            mag_tilde = self.model.decode(z)

            grid = torchvision.utils.make_grid(
                mag_tilde.reshape(-1, 1, self.n_mels, self.n_frames),
                n_sounds_generated_from_random,
                pad_value=1,
            )

            waveform_tilde = self.transform.backward(mag_tilde)
            waveform_tilde /= torch.max(abs(waveform_tilde))

        self.writer.add_image("gen/rand/melspec", grid, self.epoch)
        self.writer.add_audio(
            "gen/rand/griffinlim",
            waveform_tilde.reshape(-1),
            self.epoch,
            sample_rate=LoopDataset.FS,
        )
        return waveform_tilde, grid

    def generate_latent_explore(self, n_sounds_per_dimension):
        logger.info("exploring latent space")

        with torch.no_grad():
            z = torch.zeros(n_sounds_per_dimension * self.n_latent, self.n_latent).to(
                self.device
            )
            for i in range(self.n_latent):
                a = i * n_sounds_per_dimension
                b = (i + 1) * n_sounds_per_dimension
                z[a:b, i] = torch.linspace(-2, +2, n_sounds_per_dimension)
            mag_tilde = self.model.decode(z)
            grid = torchvision.utils.make_grid(
                mag_tilde.reshape(-1, 1, self.n_mels, self.n_frames),
                n_sounds_per_dimension,
                pad_value=1,
            )

            waveform_tilde = self.transform.backward(mag_tilde)
            waveform_tilde /= torch.max(abs(waveform_tilde))

        self.writer.add_image("gen/explo_latent/melspec", grid, self.epoch)
        self.writer.add_audio(
            "gen/explo_latent/griffinlim",
            waveform_tilde.reshape(-1),
            self.epoch,
            sample_rate=LoopDataset.FS,
        )
        return waveform_tilde, grid

    def load_state(self, path: Path):
        logger.info(f"Loading state from {path}")

        data = torch.load(path, map_location=self.device)
        self.epoch = data["epoch"]
        self.model.load_state_dict(data["model_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])
        self.beta = data["beta"]
        self.use_beta_warmup = data["use_beta_warmup"]
        self.warmup_beta_interval = data["warmup_beta_interval"]
        self.warmup_epoch_interval = data["warmup_epoch_interval"]

    def save_state(self):
        path = Path(".") / f"train_{datetime.now().isoformat('_')}_{self.epoch}.pth"
        logger.info(f"Saving state to {path}")
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "beta": self.beta,
                "use_beta_warmup": self.use_beta_warmup,
                "warmup_beta_interval": self.warmup_beta_interval,
                "warmup_epoch_interval": self.warmup_epoch_interval,
            },
            path,
        )
