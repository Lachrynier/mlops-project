"""Project training module."""
import os

# from proj.model import create_model
import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb


@hydra.main(config_path="../../configs/hydra", config_name="config", version_base=None)
def train(cfg: DictConfig):
    """Train a model on MNIST."""
    print(f"### Configuration: \n{OmegaConf.to_yaml(cfg, resolve=True)}")
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    n_classes = cfg.model.num_classes

    run = wandb.init(
        project=cfg.wandb.project,
        # config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
        entity=cfg.wandb.entity,
        job_type="train",
        config_exclude_keys=cfg.wandb
    )

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cfg.seed)

    # model = create_model(num_classes=10).to(device)
    model = instantiate(cfg.model)

    model_name = f"{cfg.model.architecture}_c{n_classes:03}"
    artifact = wandb.Artifact(
        name=model_name,
        type="Model",
        description=f"A model trained to classify {cfg.model.num_classes} classes from Caltech256.",
        metadata={"pretrained": cfg.model.pretrained}
    )

    os.makedirs("models", exist_ok=True)
    if cfg.model.pretrained:
        torch.save(model.state_dict(), f"models/{model_name}.pt")
        artifact.add_file("models/model.pt")
        run.log_artifact(artifact)
        return

    try:
        train_dataset = torch.load(f"data/processed/subset{n_classes:03}_train.pt", weights_only=False)
    except FileNotFoundError as e:
        e.strerror = f'''The dataset .pt file could not be found.\n
        Please run 'process-data --num-classes {n_classes}' from an activated python environment.'''

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    criterion = nn.CrossEntropyLoss()

    # arrays for plotting train loss and accuracy
    train_loss = []
    train_accuracy = []

    print("Training")
    model.train()

<<<<<<< HEAD
    for epoch in range(epochs):
        for images, labels in tqdm(iter(train_dataloader)):
=======
    for _ in range(cfg.epochs):
        for images, labels in tqdm(train_dataloader):

>>>>>>> 0d04bb1 (first working hydra integration in train.py)
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            train_loss.append(loss.item())

            predictions = output.argmax(dim=1)
            accuracy = (predictions == labels).float().mean().item()
            train_accuracy.append(accuracy)

            loss.backward()
            optimizer.step()

            # wandb logging
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

<<<<<<< HEAD
    # save weights and log with wandb
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")
    artifact = wandb.Artifact(
        name="Caltech256_model", type="Model", description="A model trained to classify 10 classes from Caltech256."
    )
    artifact.add_file("models/model.pth")
=======
    #save weights and log with wandb
    torch.save(model.state_dict(), "models/model.pt")
    artifact.add_file("models/model.pt")
>>>>>>> 0d04bb1 (first working hydra integration in train.py)
    run.log_artifact(artifact)

    # plot code:
    # os.makedirs("reports/figures", exist_ok=True)
    # ...


if __name__ == "__main__":
    train()
