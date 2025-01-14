import io
import tarfile
from collections.abc import Callable

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
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self._download()

        if not self._exists():
            raise RuntimeError("Dataset not found. Perhaps you forgot to use download=True?")

    def _exists(self) -> None:
        return (self.root / "256_ObjectCategories").exists()

    def _download(self) -> None:
        if self._exists():
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
            tar.extractall(self.root, filter=None)

    def __len__(self) -> int:
        """"Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = Caltech256(raw_data_path)
    dataset.preprocess(output_folder)

def main():
    Caltech256(root="data/raw", train=True, download=True)

if __name__ == "__main__":
    typer.run(main)
