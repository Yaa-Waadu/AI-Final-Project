import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path


# --------------------------------------------------
# Model architecture
# --------------------------------------------------


class ASLCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# --------------------------------------------------
# Device
# --------------------------------------------------

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# --------------------------------------------------
# Classes
# --------------------------------------------------

CLASSES = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]


# --------------------------------------------------
# Model path
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "best_asl_model2026-08-12T18_59_20.037038.pth.zip"


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = ASLCNN(num_classes=len(CLASSES))

state_dict = torch.load(MODEL_PATH, map_location=device)

model.load_state_dict(state_dict)

model.to(device)
model.eval()

print(f"ASL model loaded successfully on {device}")


# --------------------------------------------------
# Image preprocessing
# Same preprocessing used for testing
# --------------------------------------------------

test_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


# --------------------------------------------------
# Prediction function
# --------------------------------------------------


def predict_image(image_path):
    """
    Takes an image path and returns the predicted
    ASL letter and confidence.
    """

    image = Image.open(image_path)

    image = test_transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.inference_mode():
        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_letter = CLASSES[predicted_index.item()]

    confidence_percentage = confidence.item() * 100

    return {
        "prediction": predicted_letter,
        "confidence": round(confidence_percentage, 2),
    }
