from .datasets import LoopDataset
from .helpers import find_normalizer
import torch
import torchvision
import torch.utils.data


def generate_explore(
    model: torch.nn.Module,
    transform: torch.nn.Module,
    n_latent: int,
    n_sounds_per_dimension: int,
):
    with torch.no_grad():
        device = model.encoder[0].weight.device
        n_mels = transform.n_mels
        n_frames = transform.get_n_frames(65536)

        z = torch.zeros(n_sounds_per_dimension * n_latent, n_latent).to(device)
        for i in range(n_latent):
            a = i * n_sounds_per_dimension
            b = (i + 1) * n_sounds_per_dimension
            z[a:b, i] = torch.linspace(-2, +2, n_sounds_per_dimension)
        mag_tilde = model.decode(z)
        grid = torchvision.utils.make_grid(
            mag_tilde.reshape(-1, 1, n_mels, n_frames),
            n_sounds_per_dimension,
            pad_value=1,
        )

        waveform_tilde = transform.backward(mag_tilde)
        waveform_tilde /= torch.max(abs(waveform_tilde))

    return (waveform_tilde, grid)
