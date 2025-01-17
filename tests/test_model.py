import random
from proj.model import create_model
import torch

def test_model():
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = random.randint(2, 257)
    model = create_model(num_classes=num_classes).to(device)

    input = torch.randn(1, 3, 28, 28)

    model.eval()
    output = model(input)

    assert output.shape == (1, num_classes)

