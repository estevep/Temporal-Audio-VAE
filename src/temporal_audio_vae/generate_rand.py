from .datasets import LoopDataset
from .models import MelSpecVAE, construct_encoder_decoder
from .transforms import Log1pMelSpecPghi
from .helpers import find_normalizer
import torch
import logging
import torchvision
from torch.utils.tensorboard import SummaryWriter

def generate_rand(model: torch.nn.Module, 
                  transform: torch.nn.Module, 
                  n_sounds_generated_from_random: int, 
                  n_latent: int):
    
    with torch.no_grad():
        
        device = model.device
        n_mels = transform.n_mels
        n_frames = transform.get_n_frames()


        z = torch.randn(n_sounds_generated_from_random, n_latent).to(device)
        mag_tilde = model.decode(z)

        grid = torchvision.utils.make_grid(
            mag_tilde.reshape(-1, 1, n_mels, n_frames), 1
        )

        waveform_tilde = transform.backward(mag_tilde)
        waveform_tilde /= torch.max(abs(waveform_tilde))

    return(waveform_tilde, grid)



