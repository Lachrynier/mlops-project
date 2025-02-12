"""Project training module."""

import os

import hydra
import torch
import torch.nn as nn
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from google.cloud import secretmanager
from pathlib import Path


# get wandb api key with gcloud secrets
project_id = "mlops-project-77"
secret_id = "WANDB_API_KEY"
client = secretmanager.SecretManagerServiceClient()
secret_version_name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
response = client.access_secret_version(name=secret_version_name)
secret_key = response.payload.data.decode()

wandb.login(key=secret_key)


@hydra.main(config_path="../../configs/hydra", config_name="config", version_base=None)
def train(cfg: DictConfig):
    """Train a model on MNIST."""
    print(f"### Configuration: \n{OmegaConf.to_yaml(cfg, resolve=True)}")
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    num_classes = cfg.model.num_classes

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        job_type="train",
        config=config,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    model = instantiate(cfg.model).to(device)

    artifact = wandb.Artifact(
        name=cfg.model_name,
        type="Model",
        description=f"A model trained to classify {cfg.model.num_classes} classes from Caltech256.",
        metadata={"pretrained": cfg.model.pretrained},
    )

    data_dir = Path(cfg.data_dir)

    try:
        train_dataset = torch.load(data_dir / f"subset{num_classes}_train.pt", weights_only=False)
    except FileNotFoundError as e:
        e.strerror = f"""The dataset .pt file could not be found.\n
        Please run 'python src/proj/data.py --num-classes {num_classes}' from an activated python environment."""
        raise e

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    criterion = nn.CrossEntropyLoss()

    print("Training")
    model.train()

    for epoch in range(cfg.epochs):
        for images, labels in tqdm(train_dataloader, desc=f"Epoch: {epoch + 1}", disable=not cfg.print_progress):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)

            predictions = output.argmax(dim=1)
            accuracy = (predictions == labels).float().mean().item()

            loss.backward()
            optimizer.step()

            # wandb logging
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

    # save weights and log with wandb
    root = Path("./")

    if Path("/gcs/data_bucket_77").exists():
        root = Path("/gcs/data_bucket_77")

    os.makedirs(root / "models", exist_ok=True)
    torch.save(model.state_dict(), root / f"models/{cfg.model_name}.pt")
    artifact.add_file(root / f"models/{cfg.model_name}.pt")
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    train()
