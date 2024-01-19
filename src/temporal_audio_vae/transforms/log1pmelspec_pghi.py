import torch
import torchaudio
from torch import nn
from typing import Optional, Callable, Tuple
import pghipy
import numpy as np


# extended from torchaudio.transform.MelSpectrogram
class Log1pMelSpecPghi(nn.Module):
    __constants__ = [
        "sample_rate",
        "n_fft",
        "win_length",
        "hop_length",
        "pad",
        "power",
        "normalized",
        "n_mels",
        "f_max",
        "f_min",
        "norm",
        "mel_scale",
        "n_stft",
    ]

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 1024,
        hop_length: Optional[int] = 64,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 512,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        norm: Optional[str] = None,
        griffin_lim_iter=32,
        drop_last_column=True,
    ):
        super(Log1pMelSpecPghi, self).__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pad = pad
        self.normalized = normalized
        self.center = center
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = f_max
        self.f_min = f_min
        self.norm = norm
        self.n_stft = self.n_fft // 2 + 1
        self.griffin_lim_iter = griffin_lim_iter
        self.drop_last_column = drop_last_column

        self.window_np, self.gamma = pghipy.get_default_window(self.n_fft)
        self.winsynth_np = pghipy.calculate_synthesis_window(
            self.n_fft, self.hop_length, self.window_np
        )
        self.window = torch.from_numpy(self.window_np).type(torch.float32)
        self.winsynth = torch.from_numpy(self.winsynth_np).type(torch.float32)
        self.window_fn = lambda n: self.window
        self.winsynth_fn = lambda n: self.winsynth

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            pad=self.pad,
            window_fn=self.window_fn,
            power=None,  # get complex spectrogram
            normalized=self.normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=True,
        )
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            f_min=self.f_min,
            f_max=self.f_max,
            n_stft=self.n_stft,
            norm=self.norm,
            mel_scale="htk",
        )
        self.inv_mel_scale = torchaudio.transforms.InverseMelScale(
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            f_min=self.f_min,
            f_max=self.f_max,
            n_stft=self.n_stft,
            norm=self.norm,
            mel_scale="htk",
            driver="gelsd",
            # see https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html
            # safest option seems "gelsd", but does not work on CUDA
            # "gels" should be used (default), but it crashes because the matrix that goes in is not full rank. why ?
        )
        self.inv_spectrogram = torchaudio.transforms.InverseSpectrogram(
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            pad=self.pad,
            window_fn=self.winsynth_fn,
            normalized=self.normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=True,
        )

    def forward(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """stft -> mel scale -> log1p

        Returns:
            (log_mel_spectrogram, phase)
        """
        
        specgram = self.spectrogram(waveform)  # Dimension: (…, freq, time)
        mag, phase = torch.abs(specgram), torch.angle(specgram)
        mel_specgram = self.mel_scale(mag)  # dimension: (…, n_mels, time)
        log_mel_specgram = torch.log1p(mel_specgram)
        if self.drop_last_column:
            log_mel_specgram = log_mel_specgram[:, :, : log_mel_specgram.shape[2] - 1]
            phase = phase[:, :, : phase.shape[2] - 1]

        return log_mel_specgram, phase

    def backward(
        self, log1p_mel_specgram: torch.Tensor, phase: torch.Tensor = None
    ) -> torch.Tensor:
        """exp1m -> inv mel scale -> istft
        Arguments:
            log1p_mel_specgram
            phase (optional): if given, will copy phase to reconstruct the waveform
                                if not, will use griffin-lim to reconstruct phase

        Returns:
            reconstructed waveform
        """
        mel_specgram = torch.expm1(log1p_mel_specgram)

        # FIXME: tensor must be on cpu due to problem in inv_mel_scale :(
        # we force calculation on cpu and copy back to gpu
        self.inv_mel_scale = self.inv_mel_scale.cpu()
        device = mel_specgram.device
        mag = self.inv_mel_scale(mel_specgram.cpu())

        if phase is None:  # reconstruct phase
            waveform = self.__pghi_reconstruct_phase(mag).to(device)
        else:
            # Invert
            mag = mag.to(device)
            specgram = mag * torch.exp(1j * phase)
            waveform = self.inv_spectrogram(specgram)
        return waveform

    def __pghi_reconstruct_phase(self, mag: torch.Tensor) -> torch.Tensor:
        mag_np = mag.cpu().detach().numpy()
        if len(mag_np.shape) == 2:
            mag_np = mag_np.reshape(1, mag_np.shape[0], mag_np.shape[1])
        waveforms_np = []
        for i_batch in range(mag_np.shape[0]):
            m = mag_np[i_batch].T
            phase = pghipy.pghi(
                m,
                win_length=self.n_fft,
                hop_length=self.hop_length,
                gamma=self.gamma,
            )
            waveform = m * np.exp(1.0j * phase)
            if self.griffin_lim_iter > 0:
                waveform = pghipy.griffin_lim(
                    waveform,
                    win_length=self.n_fft,
                    hop_length=self.hop_length,
                    window=self.window_np,
                    synthesis_window=self.winsynth_np,
                    n_iters=self.griffin_lim_iter,
                )
            waveforms_np.append(waveform.T)
        return torch.from_numpy(np.vstack(waveforms_np))

    def get_n_frames(self, n_samples: int):
        # see https://pytorch.org/docs/stable/generated/torch.stft.html
        ret = 0
        if self.center:
            ret = 1 + n_samples // self.hop_length
        else:
            ret = 1 + (n_samples - self.n_fft) // self.hop_length
        if self.drop_last_column:
            ret -= 1
        return ret
