import torch
import lmdb
import torch.utils.data
from udls.generated import AudioExample
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


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
        assert (
            Path(db_path) / "data.mdb"
        ).exists(), f"could not find data.mdb in {db_path} !"

        self.env = lmdb.open(str(db_path), lock=False)
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

    def get_loaders(self, batch_size=128, num_threads=0, train_ratio=0.2):
        generator = torch.Generator().manual_seed(42)
        train_dataset, valid_dataset = torch.utils.data.random_split(
            self, [train_ratio, 1 - train_ratio], generator
        )
        logger.debug(
            f"splitting dataset: train={len(train_dataset)}, valid={len(valid_dataset)}"
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_threads,
            shuffle=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_threads,
        )

        return train_loader, valid_loader
