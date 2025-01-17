import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, TensorDataset, Subset, random_split

import os
import io
import tarfile
from collections.abc import Callable
from PIL import Image

import typer
import requests
from tqdm import tqdm
from pathlib import Path

class Caltech256(Dataset):
    """My custom dataset."""

    def __init__(
        self,
        root: Path | str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        self.root = Path(root)
        self.tar_path = self.root / "256_ObjectCategories.tar"
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self._download()

        if not self.tar_path.exists():
            raise RuntimeError("Dataset not found. Perhaps you forgot to use download=True?")

        self.tar = tarfile.open(self.tar_path, mode="r:")

        self.targets = []
        self.imgs = []

        for member in self.tar.getmembers():
            if not member.name.endswith(".jpg"):
                continue

            self.imgs.append(member.name)
            self.targets.append(int(member.name.split("/")[2][0:3]) - 1)

    def _download(self) -> None:
        if self.tar_path.exists():
           return

        self.root.mkdir(parents=True, exist_ok=True)

        url = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar"
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 4096

        with tqdm(desc="Downloading '256_ObjectCategories.tar'", total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(self.tar_path, "wb") as tar_file:
                for data in response.iter_content(chunk_size):
                    tar_file.write(data)
                    progress_bar.update(len(data))

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index: int):

        image = Image.open(self.tar.extractfile(self.imgs[index]))
        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

def preprocess_subset(
        root: str,
        num_classes: int | None = None,
        test_ratio = 0.2,
        download: bool = False,
    ):
    """
    num_classes: The first number of classes to be used in the subset.
                 Setting it to None chooses all classes.
    """
    if num_classes is None:
        num_classes = 257 # all classes

    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2*(x-0.5)) # Renormalize to [-1,1]
    ])

    dataset = Caltech256(root=root, transform=transform, download=download)

    # Only keep the indices for the first num_class classes
    subset = Subset(
        dataset,
        [i for i, target in enumerate(dataset.targets) if target < num_classes]
    )

    test_size = int(len(subset) * test_ratio)
    train_size = len(subset) - test_size
    train_subset, test_subset = random_split(subset, [train_size, test_size])

    train_images = torch.stack([train_subset[i][0] for i in range(len(train_subset))])
    train_labels = torch.tensor([train_subset[i][1] for i in range(len(train_subset))])

    test_images = torch.stack([test_subset[i][0] for i in range(len(test_subset))])
    test_labels = torch.tensor([test_subset[i][1] for i in range(len(test_subset))])

    os.makedirs("./data/processed", exist_ok=True)

    torch.save(
        TensorDataset(train_images, train_labels),
        f'./data/processed/subset{num_classes}_train.pt'
    )
    torch.save(
        TensorDataset(test_images, test_labels),
        f'./data/processed/subset{num_classes}_test.pt'
    )

def main(
        num_classes: int = None,
        download: bool = True
    ):
    preprocess_subset(root = "data/raw", num_classes=num_classes, download=download)

if __name__ == '__main__':
    typer.run(main)
