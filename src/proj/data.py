"""Data module of project."""

import tarfile
from collections.abc import Callable
from pathlib import Path
import re

import requests
import torch
import typer
from PIL import Image
from torch.utils.data import Dataset, Subset, TensorDataset, random_split
from torchvision import transforms
from tqdm import tqdm

TRANSFORM = transforms.Compose(
    [
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2 * (x - 0.5)),  # Renormalize to [-1,1]
    ]
)


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

        self.num_classes = len(set(self.targets))

    def _download(self) -> None:
        if self.tar_path.exists():
            return

        self.root.mkdir(parents=True, exist_ok=True)

        url = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar"
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 4096

        with tqdm(
            desc="Downloading '256_ObjectCategories.tar'", total=total_size, unit="B", unit_scale=True
        ) as progress:
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

    def class_names(self) -> list[str]:
        class_names = 257 * [None]

        pattern = "\\.([a-z0-9-]*)\\/."
        for img, target in zip(self.imgs, self.targets):
            if class_names[target] is not None:
                continue
            match = next(re.finditer(pattern, img))
            class_names[target] = match.group(1)

        return class_names


def preprocess_subset(
    raw_dir: Path | str,
    processed_dir: Path | str,
    num_classes: int | None = None,
    test_ratio: float = 0.2,
    download: bool = False,
):
    """
    num_classes: The first number of classes to be used in the subset.
                 Setting it to None chooses all classes.
    """

    dataset = Caltech256(root=raw_dir, transform=TRANSFORM, download=download)

    if num_classes is None:
        num_classes = dataset.num_classes  # all classes

    # Only keep the indices for the first num_class classes
    subset = Subset(dataset, [i for i, target in enumerate(dataset.targets) if target < num_classes])

    train_subset, test_subset = random_split(subset, [1 - test_ratio, test_ratio])

    print("Constructing training set...")
    train_subset = tuple(zip(*train_subset, strict=True))
    train_images = torch.stack(train_subset[0])
    train_labels = torch.tensor(train_subset[1])
    print("Constructing test set...")
    test_subset = tuple(zip(*test_subset, strict=True))
    test_images = torch.stack(test_subset[0])
    test_labels = torch.tensor(test_subset[1])

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    torch.save(TensorDataset(train_images, train_labels), processed_dir / f"subset{num_classes}_train.pt")
    torch.save(TensorDataset(test_images, test_labels), processed_dir / f"subset{num_classes}_test.pt")


def main(num_classes: int = None, download: bool = True, raw_dir: str = "data/raw"):
    """Preprocess dataset."""
    preprocess_subset(raw_dir=raw_dir, processed_dir="data/processed", num_classes=num_classes, download=download)


if __name__ == "__main__":
    typer.run(main)
