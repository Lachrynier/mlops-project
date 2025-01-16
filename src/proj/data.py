import io
import tarfile
from collections.abc import Callable
from PIL import Image

import typer
import requests
from tqdm import tqdm
from pathlib import Path

from torch.utils.data import Dataset

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

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = Caltech256(raw_data_path)
    dataset.preprocess(output_folder)

def main(root: Path = Path("data/raw")):
    dataset = Caltech256(root=root, download=True)

if __name__ == "__main__":
    typer.run(main)
