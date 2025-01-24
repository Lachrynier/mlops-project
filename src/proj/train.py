"""Project training module."""

import os

# from proj.model import create_model
import hydra
import torch
import torch.nn as nn
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from google.cloud import secretmanager

from proj.data import Caltech256, TRANSFORM


#get wandb api key with gcloud secrets
project_id = "mlops-project-77"
secret_id = "WANDB_API_KEY"
client = secretmanager.SecretManagerServiceClient()
secret_version_name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
response = client.access_secret_version(name=secret_version_name)
secret_key = response.payload.data.decode()

wandb.login(key = secret_key)

@hydra.main(config_path="../../configs/hydra", config_name="config", version_base=None)
def train(cfg: DictConfig):
    """Train a model on MNIST."""
    print(f"### Configuration: \n{OmegaConf.to_yaml(cfg, resolve=True)}")
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    num_classes = cfg.model.num_classes

    run = wandb.init(
        project=cfg.wandb.project,
        # config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
        entity=cfg.wandb.entity,
        job_type="train",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    # model = create_model(num_classes=10).to(device)
    model = instantiate(cfg.model)

    model_name = f"{cfg.model.architecture}_c{num_classes}"
    artifact = wandb.Artifact(
        name=model_name,
        type="Model",
        description=f"A model trained to classify {cfg.model.num_classes} classes from Caltech256.",
        metadata={"pretrained": cfg.model.pretrained},
    )


    # For training locally with .pt files
    # try:
    #     train_dataset = torch.load(f"data/processed/subset{num_classes}_train.pt", weights_only=False)
    # except FileNotFoundError as e:
    #     e.strerror = f"""The dataset .pt file could not be found.\n
    #     Please run 'python src/proj/data.py --num-classes {num_classes}' from an activated python environment."""
    #     raise e

    # For training with vertex AI in GCP with .tar file
    train_dataset = Caltech256('gcs/data_bucket_77/data/raw', transform=TRANSFORM)


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    criterion = nn.CrossEntropyLoss()

    print("Training")
    model.train()

    for epoch in range(cfg.epochs):
        for images, labels in tqdm(train_dataloader, desc=f"Epoch: {epoch + 1}"):
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
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), f"models/{model_name}.pt")
    artifact.add_file(f"models/{model_name}.pt")
    run.log_artifact(artifact)

    # Save weights in gcloud storage
    torch.save(model.state_dict(), 'gcs/data_bucket_77/models/model_test2.pth')

if __name__ == "__main__":
    train()
