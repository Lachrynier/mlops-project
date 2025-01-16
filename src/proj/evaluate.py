import torch
from tqdm import tqdm
from model import create_model


def evaluate(model_checkpoint: str):
    print("Evaluating model")

    batch_size = 32

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model(num_classes=20).to(device)
    model.load_state_dict(torch.load(model_checkpoint, weights_only=True))

    test_dataset = torch.load("data/processed/subset_test.pt")
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    model.eval()
    correct = 0
    total = 0
    for images, labels in tqdm(iter(test_dataloader)):
        images, labels = images.to(device), labels.to(device)

        output = model(images)

        predictions = output.argmax(dim=1)
        correct += (predictions == labels).float().sum().item()
        total += labels.size(0)

    test_accuracy = correct / total
    print(f"Test accuracy: {test_accuracy}")

if __name__ == "__main__":
    evaluate("models/model.pth")