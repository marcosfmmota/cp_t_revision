from torch import nn
import random
from loguru import logger


class SymmetricNoise(nn.Module):
    """Applies symmetric noise transformation on label. Producing a noise transition matrix of
    1 - noise_level for clean class and noise_level / (n_classes - 1) for noisy class.
    """

    def __init__(
        self, noise_level: float, n_classes: int, seed: int | None = None, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.noise_level = noise_level
        self.n_classes = n_classes
        random.seed(seed)

    def forward(self, label):
        if random.uniform(0, 1) <= self.noise_level:
            choice_list = list(range(self.n_classes))
            try:
                choice_list.remove(label)
            except ValueError:
                logger.error(
                    "Label not found in noise switch list. Check if label enconding in [0..n_classes] range."
                )
                raise ValueError
            label = random.choice(choice_list)
        return label


class PairFlipNoise(nn.Module):
    """Applies pair flip noise transformation on label. Producing a noise transition matrix of
    1 - noise_level for clean j class and noise_level for noisy j + 1 class. In the case of clean
    label j = n_classes - 1 the noisy label is 0 (circular fashion).
    """

    def __init__(
        self, noise_level: float, n_classes: int, seed: int | None = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.noise_level = noise_level
        self.n_classes = n_classes
        random.seed(seed)

    def forward(self, label):
        if random.uniform(0, 1) <= self.noise_level:
            if label < self.n_classes - 1:
                label = label + 1
            elif label == self.n_classes - 1:
                label = 0
            else:
                logger.error(
                    "Label not found in noise switch list. Check if label enconding in [0..n_classes] range."
                )
                raise ValueError
        return label
