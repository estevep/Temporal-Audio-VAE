from .datasets import LoopDataset
from .helpers import find_normalizer
import torch
import torchvision
import torch.utils.data


def generate_data(model: torch.nn.Module, 
                  transform: torch.nn.Module, 
                  valid_loader: torch.utils.data.DataLoader,
                  n_sounds_generated_from_dataset: int
                  ):
    
    with torch.no_grad():
        
        device = model.device
        n_mels = transform.n_mels
        n_frames = transform.get_n_frames()
        valid_norm = find_normalizer(valid_loader, "valid", transform).to(device)



        waveform = next(iter(valid_loader)).to(device)
        mag, phase = transform.forward(
            waveform[:n_sounds_generated_from_dataset]
        )
        mag = valid_norm(mag)

        mag_tilde, _ = model(mag)

        waveform_tilde_copyphase = transform.backward(mag_tilde, phase)
        waveform_tilde_griffinlim = transform.backward(mag_tilde)

        grid = torchvision.utils.make_grid(
            mag_tilde.reshape(-1, 1, n_mels, n_frames), 1
        )

        waveform_tilde_copyphase /= torch.max(abs(waveform_tilde_copyphase))
        waveform_tilde_griffinlim /= torch.max(abs(waveform_tilde_griffinlim))



        
    return(waveform_tilde_copyphase, waveform_tilde_griffinlim, grid)


