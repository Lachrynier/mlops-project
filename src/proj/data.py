"""Data module of project."""
import os
import tarfile
from collections.abc import Callable
from pathlib import Path
import requests
import typer
from tqdm import tqdm

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, TensorDataset, random_split
from torchvision import transforms
import lmdb

class Caltech256(Dataset):
    """Custom Dataset class for the Caltech256 dataset."""

    def __init__(
        self,
        root: Path | str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        """Initialize dataset."""
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

        with tqdm(desc="Downloading '256_ObjectCategories.tar'", total=total_size, unit="B", unit_scale=True) as progress:
            with open(self.tar_path, "wb") as tar_file:
                for data in response.iter_content(chunk_size):
                    tar_file.write(data)
                    progress.update(len(data))

    def __len__(self) -> int:
        """Length of dataset."""
        return len(self.imgs)

    def __getitem__(self, index: int):
        """Get item from dataset."""
        image = Image.open(self.tar.extractfile(self.imgs[index]))
        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

def dataset_to_tensors(dataset: torch.utils.data.Dataset):

    images = torch.stack([image for image, _ in dataset])
    labels = torch.stack([label for _, label in dataset])

    return images, labels


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
    ])

    dataset = Caltech256(root=root, transform=transform, download=download)
     # Only keep the indices for the first num_class classes
    subset = Subset(
        dataset,
        [i for i, target in enumerate(dataset.targets) if target < num_classes]
    )
    train_subset, test_subset = random_split(subset, [1 - test_ratio, test_ratio])

    train_names = num_classes * [0]
    for image, target in train_subset:
        os.makedirs(f"data/processed/train/{target}", exist_ok=True)
        image.save(f"data/processed/train/{target}/{train_names[target]}.png")
        train_names[target] += 1
    
    
    test_names = num_classes * [0]
    for image, target in test_subset:
        os.makedirs(f"data/processed/test/{target}", exist_ok=True)
        image.save(f"data/processed/test/{target}/{test_names[target]}.png")
        test_names[target] += 1


def main(
        num_classes: int = None,
        download: bool = True
    ):
    """Preprocess dataset."""
    preprocess_subset(root = "data/raw", num_classes=num_classes, download=download)

if __name__ == '__main__':
    typer.run(main)
