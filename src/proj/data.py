import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, Subset, random_split

import os
from pathlib import Path

def preprocess_subset(num_classes=10, file_name='subset', test_ratio=0.2):
    """
    num_classes: The first number of classes to be used in the subset.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2*(x-0.5)) # Renormalize to [-1,1]
    ])

    dataset = datasets.ImageFolder(
        root='./data/raw/caltech256/256_ObjectCategories',
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
        f'./data/processed/{file_name}_train.pt'
    )
    torch.save(
        TensorDataset(test_images, test_labels),
        f'./data/processed/{file_name}_test.pt'
    )

def preprocess_full():
    preprocess_subset(num_classes=257, file_name='dataset')

if __name__ == '__main__':
    preprocess_subset(num_classes=10)