from .datasets import LoopDataset
from .models import MelSpecVAE, construct_encoder_decoder
from .transforms import Log1pMelSpecPghi
from .helpers import find_normalizer
import torch
import logging
import torchvision
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def train(dataset_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: %s", device)

    # load dataset
    dataset = LoopDataset(dataset_path)
    train_loader, valid_loader = dataset.get_loaders()
    

    # hyperparameters
    n_epochs = 50
    n_latent = 16
    n_hidden = 128
    n_mels = 512
    n_fft = 1024
    griffin_lim_iter = 64
    hop_length = 256
    beta_interval = (0, 1)  # min, max
    beta_epoch_interval = (100, 400)  # start, end
    generate_every_nth_epoch = 10
    evaluate_every_nth_epoch = 10
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
    train_norm = find_normalizer(train_loader, "train", transform).to(device)
    valid_norm = find_normalizer(valid_loader, "valid", transform).to(device)

    print(next(iter(train_loader)).size())
    print(transform(next(iter(train_loader)))[0].size())

    # Construct encoder and decoder
    encoder, decoder = construct_encoder_decoder(n_hidden=n_hidden, n_latent=n_latent)
    # Build the VAE model
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

    for epoch in range(1, n_epochs + 1):
        print("Epoch = {}".format(epoch))
        # TRAINING
        model.train()
        logger.info("training")

        # beta = beta_warmup(epoch, beta_interval, beta_epoch_interval)
        beta = 0

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

        model.eval()
        logger.info("evaluation")

        full_loss = 0
        recons_loss = 0
        kl_div = 0

        ## EVALUATION
        if epoch % evaluate_every_nth_epoch == 0 or epoch == n_epochs - 1:
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

        if epoch % generate_every_nth_epoch == 0 or epoch == n_epochs - 1:
            logger.info("generating from dataset")
            with torch.no_grad():
                waveform = next(iter(valid_loader)).to(device)
                mag, phase = transform.forward(
                    waveform[:n_sounds_generated_from_dataset]
                )
                mag = train_norm(mag)

                mag_tilde, _ = model(mag)

                waveform_tilde_copyphase = transform.backward(mag_tilde, phase)
                waveform_tilde_griffinlim = transform.backward(mag_tilde)

                grid = torchvision.utils.make_grid(
                    mag_tilde.reshape(-1, 1, n_mels, n_frames), 1
                )
                WRITER.add_image("gen/dataset/melspec", grid, epoch)

                waveform_tilde_copyphase /= torch.max(abs(waveform_tilde_copyphase))
                waveform_tilde_griffinlim /= torch.max(abs(waveform_tilde_griffinlim))

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
            with torch.no_grad():
                z = torch.randn(n_sounds_generated_from_random, n_latent).to(device)
                mag_tilde = model.decode(z)

                grid = torchvision.utils.make_grid(
                    mag_tilde.reshape(-1, 1, n_mels, n_frames), 1
                )
                WRITER.add_image("gen/rand_latent/melspec", grid, epoch)

                waveform_tilde = transform.backward(mag_tilde)
                waveform_tilde /= torch.max(abs(waveform_tilde))

                WRITER.add_audio(
                    "gen/rand_latent/griffinlim",
                    waveform_tilde_griffinlim.reshape(-1),
                    epoch,
                    sample_rate=LoopDataset.FS,
                )
            logger.info("exploring latent space")
            with torch.no_grad():
                n_sounds_per_dimension = 5
                z = torch.zeros(n_sounds_per_dimension * n_latent, n_latent).to(device)
                for i in range(n_latent):
                    a = i * n_sounds_per_dimension
                    b = (i + 1) * n_sounds_per_dimension
                    z[a:b, i] = torch.linspace(-2, +2, n_sounds_per_dimension)
                mag_tilde = model.decode(z)
                grid = torchvision.utils.make_grid(
                    mag_tilde.reshape(-1, 1, n_mels, n_frames), n_sounds_per_dimension
                )
                WRITER.add_image("gen/explo_latent/melspec", grid, epoch)
                waveform_tilde = transform.backward(mag_tilde)
                waveform_tilde /= torch.max(abs(waveform_tilde))

                WRITER.add_audio(
                    "gen/explo_latent/griffinlim",
                    waveform_tilde_griffinlim.reshape(-1),
                    epoch,
                    sample_rate=LoopDataset.FS,
                )
