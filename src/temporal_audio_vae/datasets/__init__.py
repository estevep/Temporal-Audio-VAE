import torch
import lmdb
from udls.generated import AudioExample


class LoopDataset(torch.utils.data.Dataset):
    FS = 44100
    LEN_SAMPLES = 65536

    def __init__(self, db_path: str) -> None:
        """Audio loops dataset

        Each example is 65536 samples long and is sampled at 44100 Hz

        Args:
            db_path (str): Path to the directory containing `data.mdb`
        """
        super().__init__()

        self.env = lmdb.open(db_path, lock=False)
        with self.env.begin(write=False) as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        with self.env.begin(write=False) as txn:
            ae = AudioExample.FromString(txn.get(self.keys[idx]))

        buffer = ae.buffers["waveform"]
        assert buffer.precision == AudioExample.Precision.INT16
        assert buffer.sampling_rate == self.FS

        audio = torch.frombuffer(buffer.data, dtype=torch.int16)
        audio = audio.float() / (2**15 - 1)
        assert len(audio) == self.LEN_SAMPLES

        return audio
