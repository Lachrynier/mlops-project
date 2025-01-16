import torch
import torch.nn as nn

from tqdm import tqdm
import os

from model import create_model


def train(lr: float=1e-3):

    batch_size = 32
    epochs = 3

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = torch.load("data/processed/subset_train.pt")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    model = create_model(num_classes=20).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    #arrays for plotting train loss and accuracy
    train_loss = []
    train_accuracy = []

    print("Training")
    model.train()

    for epoch in range(epochs):
        for images, labels in tqdm(iter(train_dataloader)):
            
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

    #save weights
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pth")

    #plot code:
    # os.makedirs("reports/figures", exist_ok=True)
    # ...
    

if __name__ == "__main__":
    train()