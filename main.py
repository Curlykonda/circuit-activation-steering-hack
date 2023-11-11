import logging
import random

import numpy as np
import torch

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S",
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)

_DEFAULT_RANDOM_SEED = 21


def _seed_everything(seed=_DEFAULT_RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    logger.info("Hello World!")
