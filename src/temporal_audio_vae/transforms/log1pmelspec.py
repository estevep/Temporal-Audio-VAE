import torch
import torchaudio
from torch import nn
from typing import Optional, Callable, Tuple


# extended from torchaudio.transform.MelSpectrogram
class Log1pMelSpec(nn.Module):
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
        win_length: Optional[int] = 128,
        hop_length: Optional[int] = 64,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 512,
        window_fn: Callable[..., torch.Tensor] = torch.hann_window,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        norm: Optional[str] = None,
        griffin_lim_iter=32,
        griffin_lim_momentum=0.99,
    ):
        super(Log1pMelSpec, self).__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.normalized = normalized
        self.center = center
        self.n_mels = n_mels  # number of mel frequency bins
        self.f_max = f_max
        self.f_min = f_min
        self.norm = norm
        self.n_stft = self.n_fft // 2 + 1

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            window_fn=window_fn,
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
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            window_fn=window_fn,
            normalized=self.normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=True,
        )
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.n_fft,
            n_iter=griffin_lim_iter,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window_fn=window_fn,
            power=1,  # magnitude spectrogram
            wkwargs=wkwargs,
            momentum=griffin_lim_momentum,
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
        mag = mag.to(device)

        if phase is None:  # reconstruct phase
            # TODO: use other (better) reconstruction methods ?
            waveform = self.griffin_lim(mag)
        else:  # copy original phase
            specgram = mag * torch.exp(1j * phase)
            waveform = self.inv_spectrogram(specgram)
        return waveform

    def get_n_frames(self, n_samples: int):
        # see https://pytorch.org/docs/stable/generated/torch.stft.html
        if self.center:
            return 1 + n_samples // self.hop_length
        else:
            1 + (n_samples - self.n_fft) // self.hop_length