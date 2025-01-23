import torch
from tqdm import tqdm
from proj.model import create_model

import hydra


@hydra.main(config_path="../../configs/hydra", config_name="config", version_base=None)
def evaluate(cfg):
    print("Evaluating model")

    # batch_size = 32
    batch_size = cfg.eval.batch_size
    num_classes = cfg.model.num_classes

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = create_model(num_classes=num_classes).to(device)
    model_name = f"{cfg.model.architecture}_c{num_classes}"

    model.load_state_dict(torch.load(f"models/{model_name}.pt", weights_only=True))

    test_dataset = torch.load("data/processed/subset10_test.pt", weights_only=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model.eval()
    correct = 0
    total = 0
    for images, labels in tqdm(test_dataloader):
        images, labels = images.to(device), labels.to(device)

        output = model(images)

        predictions = output.argmax(dim=1)
        correct += (predictions == labels).float().sum().item()
        total += labels.size(0)

    test_accuracy = correct / total
    print(f"Test accuracy: {test_accuracy}")


if __name__ == "__main__":
    evaluate()
