"""Project model module."""

import hydra
import timm
import torch
from hydra.utils import instantiate


# redundant?
def create_model(architecture="resnet50.a1_in1k", pretrained=False, num_classes=257):
    """Create model instance."""
    model = timm.create_model(architecture, pretrained=pretrained, num_classes=num_classes)
    return model


@hydra.main(config_path="../../configs/hydra", config_name="model", version_base=None)
def model_params(cfg):
    """Print model parameters."""
    model = instantiate(cfg.model)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    model_params()
