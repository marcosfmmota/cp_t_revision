from abc import ABC
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from PIL import Image
from cp_t_revision.cifar_n.datasets import input_dataset


class LitDataModule(LightningDataModule, ABC):
    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, num_workers=self.num_workers)


class MNISTDataModule(LitDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 32,
        val_proportion: list[int | float] = [55_000, 5_000],
        noise_transform: nn.Module | None = None,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.val_proportion = val_proportion
        self.noise_transform = noise_transform
        self.num_workers = num_workers

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        split_generator = torch.Generator().manual_seed(123)
        if stage == "fit":
            mnist_full = MNIST(
                self.data_dir,
                train=True,
                transform=self.transforms,
                target_transform=self.noise_transform,
            )
            self.train, self.val = torch.utils.data.random_split(
                mnist_full, self.val_proportion, generator=split_generator
            )
        if stage == "test":
            self.test = MNIST(self.data_dir, train=False, transform=self.transforms)

        if stage == "predict":
            self.predict = MNIST(self.data_dir, train=False, transform=self.transforms)


class CIFAR10DataModule(LitDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 32,
        val_proportion: list[int | float] = [40_000, 10_000],
        train_transforms: torch.nn.Module = transforms.ToTensor(),
        test_transforms: torch.nn.Module = transforms.ToTensor(),
        noise_transform: nn.Module | None = None,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.val_proportion = val_proportion
        self.noise_transform = noise_transform
        self.num_workers = num_workers

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage):
        split_generator = torch.Generator().manual_seed(123)
        if stage == "fit":
            cifar10_full = CIFAR10(
                self.data_dir,
                train=True,
                transform=self.train_transforms,
                target_transform=self.noise_transform,
            )
            self.train, self.val = torch.utils.data.random_split(
                cifar10_full, self.val_proportion, generator=split_generator
            )
        if stage == "test":
            self.test = CIFAR10(self.data_dir, train=False, transform=self.test_transforms)

        if stage == "predict":
            self.predict = CIFAR10(self.data_dir, train=False, transform=self.test_transforms)


class CIFAR100DataModule(LitDataModule):
    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 32,
        val_proportion: list[int | float] = [40_000, 10_000],
        train_transforms: torch.nn.Module = transforms.ToTensor(),
        test_transforms: torch.nn.Module = transforms.ToTensor(),
        noise_transform: nn.Module | None = None,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.val_proportion = val_proportion
        self.noise_transform = noise_transform
        self.num_workers = num_workers

    def prepare_data(self):
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage):
        split_generator = torch.Generator().manual_seed(123)
        if stage == "fit":
            cifar100_full = CIFAR100(
                self.data_dir,
                train=True,
                transform=self.train_transforms,
                target_transform=self.noise_transform,
            )
            self.train, self.val = torch.utils.data.random_split(
                cifar100_full, self.val_proportion, generator=split_generator
            )
        if stage == "test":
            self.test = CIFAR100(self.data_dir, train=False, transform=self.test_transforms)

        if stage == "predict":
            self.predict = CIFAR100(self.data_dir, train=False, transform=self.test_transforms)


class Clothing1M(Dataset):
    def __init__(
        self,
        annotations_df,
        transform=None,
        data_dir: str | Path = "data/clothing1M",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.annotations_df = annotations_df
        self.transform = transform

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, index):
        img_path = self.data_dir / self.annotations_df.iloc[index, 0]
        image = Image.open(img_path)
        label = self.annotations_df.iloc[index, 1]
        if self.transform:
            image = self.transform(image)

        return image, label


class Clothing1MDataModule(LitDataModule):
    def __init__(
        self,
        data_dir=Path("data/clothing1M"),
        clean_run=False,
        batch_size=128,
        num_workers: int = 4,
        train_transforms: torch.nn.Module = transforms.ToTensor(),
        test_transforms: torch.nn.Module = transforms.ToTensor(),
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.clean_run = clean_run
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.num_workers = num_workers
        self.clean_train_k = pd.read_csv(
            data_dir / "clean_train_key_list.txt", sep=" ", header=None
        )
        self.clean_val_k = pd.read_csv(data_dir / "clean_val_key_list.txt", sep=" ", header=None)
        self.clean_test_k = pd.read_csv(data_dir / "clean_test_key_list.txt", sep=" ", header=None)
        self.noisy_train_k = pd.read_csv(
            data_dir / "noisy_train_key_list.txt", sep=" ", header=None
        )
        self.clean_all_kv = pd.read_csv(data_dir / "clean_label_kv.txt", sep=" ", header=None)
        self.noisy_all_kv = pd.read_csv(data_dir / "noisy_label_kv.txt", sep=" ", header=None)

        # Get clean image label pairs
        self.clean_train = self.clean_all_kv[self.clean_all_kv[0].isin(self.clean_train_k[0])]
        self.clean_val = self.clean_all_kv[self.clean_all_kv[0].isin(self.clean_val_k[0])]
        self.clean_test = self.clean_all_kv[self.clean_all_kv[0].isin(self.clean_test_k[0])]

        # Get noisy train label pairs
        self.noisy_train = self.noisy_all_kv[self.noisy_all_kv[0].isin(self.noisy_train_k[0])]

    def setup(self, stage):
        if stage == "fit":
            if self.clean_run:
                self.train = Clothing1M(self.clean_train, self.train_transforms, self.data_dir)
            else:
                self.train = Clothing1M(self.noisy_train, self.train_transforms, self.data_dir)
            self.val = Clothing1M(self.clean_val, self.test_transforms, self.data_dir)

        if stage == "test":
            self.test = Clothing1M(self.clean_test, self.test_transforms, self.data_dir)

        if stage == "predict":
            self.predict = Clothing1M(self.clean_test, self.test_transforms, self.data_dir)


class CIFARNDataModule(LitDataModule):
    """DataModule that uses CIFAR-10N / CIFAR-100N style noisy datasets via `input_dataset`.

    Args:
        dataset (str): 'cifar10' or 'cifar100' to select the dataset variant.
        noise_type (str | None): noise type identifier consumed by `input_dataset` (e.g. 'aggre_label', 'worse_label').
        noise_path (str | None): path to noise files if required by `input_dataset`.
        is_human (bool): whether the noise is a human-labelled CIFAR-N (affects `input_dataset`).
        data_dir (str): root directory for CIFAR data (kept for compatibility but `input_dataset` currently uses './').
        batch_size (int): batch size for loaders.
        val_proportion (list[int] | float): either a list of two ints [n_train, n_val] or a float fraction for validation split.
        num_workers (int): dataloader workers.
    """

    def __init__(
        self,
        dataset: str = "cifar10",
        noise_type: str | None = None,
        noise_path: str | None = None,
        is_human: bool = True,
        data_dir: str = "data",
        batch_size: int = 32,
        val_proportion: list[int | float] | float = [40_000, 10_000],
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_path = noise_path
        self.is_human = is_human
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_proportion = val_proportion
        self.num_workers = num_workers

        # Attributes populated in setup
        self.num_classes = None

    def setup(self, stage: str | None = None):
        # Load datasets via helper which returns already-transformed datasets
        train_dataset, test_dataset, num_classes, num_training_samples = input_dataset(
            self.dataset, self.noise_type, self.noise_path, self.is_human
        )
        self.num_classes = num_classes

        # Determine train/val split
        if isinstance(self.val_proportion, float):
            n_val = int(self.val_proportion * num_training_samples)
            n_train = num_training_samples - n_val
            split_sizes = [n_train, n_val]
        elif isinstance(self.val_proportion, (list, tuple)) and len(self.val_proportion) == 2:
            split_sizes = self.val_proportion
        else:
            raise ValueError("val_proportion must be a float or a list/tuple of two ints")

        generator = torch.Generator().manual_seed(123)

        if stage == "fit" or stage is None:
            self.train, self.val = torch.utils.data.random_split(
                train_dataset, split_sizes, generator=generator
            )

        if stage == "test" or stage is None:
            self.test = test_dataset

        if stage == "predict":
            self.predict = test_dataset

    # The LitDataModule base provides train/val/test/predict loader implementations
