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
        self.data_dir = self.root / "256_ObjectCategories"
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self._download()

        if not self.data_dir.exists():
            raise RuntimeError("Dataset not found. Perhaps you forgot to use download=True?")

        self.image_paths = [image_file for category_dir in self.data_dir.iterdir() for image_file in category_dir.iterdir() if image_file.suffix == ".jpg"]

    def _download(self) -> None:
        if self.data_dir.exists():
           return

        url = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar"
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 4096

        # For some reason tarfile does not like getting the bytes directly, so use io.BytesIO to pretend it is a file
        buffer = io.BytesIO()

        with tqdm(desc="Downloading '256_ObjectCategories.tar'", total=total_size, unit="B", unit_scale=True) as progress_bar:
            for data in response.iter_content(chunk_size):
                buffer.write(data)
                progress_bar.update(len(data))

        buffer.seek(0)

        with tarfile.open(fileobj=buffer, mode="r:") as tar:
            tar.extractall(self.root, filter='data')

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]

        image = Image.open(image_path)
        target = int(image_path.parent.name[0:3]) - 1

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

def preprocess_subset(
        num_classes: int | str = 'full',
        test_ratio=0.2
    ):
    """
    num_classes: The first number of classes to be used in the subset.
                 Setting it to 'full' chooses all classes.
    """
    if isinstance(num_classes, str):
        num_classes = 257 # all classes
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2*(x-0.5)) # Renormalize to [-1,1]
    ])

    dataset = datasets.ImageFolder(
        root='./data/raw/256_ObjectCategories',
        transform=transform
    )
    
    # Only keep the indices for the first num_class classes
    subset = Subset(
        dataset,
        [i for i in dataset.targets if i < num_classes]
    )

    test_size = int(len(subset) * test_ratio)
    train_size = len(subset) - test_size
    train_subset, test_subset = random_split(subset, [train_size, test_size])

    train_images = torch.stack([train_subset[i][0] for i in range(len(train_subset))])
    train_labels = torch.tensor([train_subset[i][1] for i in range(len(train_subset))])

    test_images = torch.stack([test_subset[i][0] for i in range(len(test_subset))])
    test_labels = torch.tensor([test_subset[i][1] for i in range(len(test_subset))])

    torch.save(
        TensorDataset(train_images, train_labels),
        f'./data/processed/subset{num_classes}_train.pt'
    )
    torch.save(
        TensorDataset(test_images, test_labels),
        f'./data/processed/subset{num_classes}_test.pt'
    )

def main(
        num_classes: int | str = 10,
        root: Path = Path("data/raw")
    ):
    Caltech256(root=root, download=True)
    preprocess_subset(num_classes=num_classes)

if __name__ == '__main__':
    typer.run(main)