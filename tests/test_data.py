import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

from proj.data import Caltech256


N_IMAGES = 10
N_CLASSES = 5


def test_caltech256_load():
    """Test the Caltech256 class."""
    dataset = Caltech256(root="data/raw_test")
    assert isinstance(dataset, Dataset)
    assert len(dataset) == N_IMAGES

    for image, target in dataset:
        assert isinstance(image, Image.Image)
        assert isinstance(target, int)
        assert target >= 0 and target < 5


def test_caltech256_transform():
    width = 28
    height = 28

    transform = transforms.Compose(
        [
            transforms.Resize([height, width]),
            transforms.ToTensor(),
        ]
    )

    dataset = Caltech256(root="data/raw_test", transform=transform)

    for image, _ in dataset:
        assert isinstance(image, torch.Tensor)
        assert image.ndim == 3
        assert image.size(2) == width
        assert image.size(1) == height


def test_caltech256_target_transform():
    dataset = Caltech256(root="data/raw_test", target_transform=(lambda x: x + 5))

    for _, target in dataset:
        assert isinstance(target, int)
        assert target >= 5 and target < 10
