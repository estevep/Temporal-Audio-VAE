import os
import torch
import torch.utils.data
import torch.nn as nn
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class func2module(nn.Module):
    def __init__(self, f):
        super(func2module, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


def find_normalizer(
    dataloader: torch.utils.data.DataLoader, name: str, transform: nn.Module
):
    assert name != ""
    if not os.path.isfile("min_" + name):
        minval, maxval = float("inf"), float("-inf")
        logger.info(
            f"Could not find normalizer for '{name}', recalculating min and max..."
        )
        for x in tqdm(dataloader):
            mag, _ = transform(x)
            minval = min(minval, mag.min())
            maxval = max(maxval, mag.max())
        torch.save(minval, "min_" + name)
        torch.save(maxval, "max_" + name)
    else:
        minval = torch.load("min_" + name)
        maxval = torch.load("max_" + name)

    logger.debug(f"normalization for {name}: {minval=:.2f}, {maxval=:.2f}")
    norm = func2module(lambda x: (x - minval) / (maxval - minval))

    return norm
