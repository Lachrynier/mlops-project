import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

from proj.data import Caltech256


N_IMAGES = 30607

def test_caltech256_load():
    """Test the Caltech256 class."""
    dataset = Caltech256(root="data/raw", download=True)
    assert isinstance(dataset, Dataset)
    assert len(dataset) == N_IMAGES

    for image, target in dataset:
        assert isinstance(image, Image.Image)
        assert type(target) == int
        assert target >= 0 and target <= 256 # not a mistake, there are 257 categories (includes "clutter")

def test_caltech256_transform():

    width = 28
    height = 28

    transform = transforms.Compose([
        transforms.Resize([height, width]),
        transforms.ToTensor(),
    ])

    dataset = Caltech256(root="data/raw", transform=transform, download=True)

    for image, _ in dataset:
        assert type(image) == torch.Tensor
        assert image.ndim == 3
        assert image.size(2) == width
        assert image.size(1) == height


def test_caltech256_target_transform():

    target_transform = lambda x: x + 256

    dataset = Caltech256(root="data/raw", target_transform=target_transform, download=True)

    for _, target in dataset:
        assert type(target) == int
        assert target >= 256 and target <= 512
