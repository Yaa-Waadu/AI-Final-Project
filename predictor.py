from pathlib import Path

import numpy as np
import torch
from PIL import Image
from rembg import new_session, remove
from torch import nn
from torchvision import transforms


class NoHandDetectedError(Exception):
    """Raised when no hand can be found in the uploaded image."""


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


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


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


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "best_asl_model2026-08-12T18_59_20.037038.pth.zip"


model = ASLCNN(num_classes=len(CLASSES))

state_dict = torch.load(MODEL_PATH, map_location=device)

model.load_state_dict(state_dict)

model.to(device)
model.eval()

print(f"ASL model loaded successfully on {device}")


# Segmentation
segmentation_session = new_session("u2netp")

ALPHA_THRESHOLD = 128
CROP_PADDING_RATIO = 0.05
MIN_SUBJECT_AREA_RATIO = 0.02


def segment_and_crop(image):
    rgba = remove(image.convert("RGB"), session=segmentation_session)

    alpha = np.array(rgba.split()[-1])
    ys, xs = np.where(alpha > ALPHA_THRESHOLD)

    if len(xs) < MIN_SUBJECT_AREA_RATIO * alpha.size:
        raise NoHandDetectedError(
            "No hand detected in the uploaded image. Try a clearer photo "
            "with the hand filling more of the frame."
        )

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    pad_x = int((x_max - x_min) * CROP_PADDING_RATIO)
    pad_y = int((y_max - y_min) * CROP_PADDING_RATIO)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(rgba.width, x_max + pad_x)
    y_max = min(rgba.height, y_max + pad_y)

    cropped = rgba.crop((x_min, y_min, x_max, y_max))

    side = max(cropped.width, cropped.height)
    black_background = Image.new("RGB", (side, side), (0, 0, 0))
    offset = ((side - cropped.width) // 2, (side - cropped.height) // 2)
    black_background.paste(cropped, offset, mask=cropped.split()[-1])

    return black_background


test_transform = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


def predict_image(image_path):
    """
    Takes an image path and returns the predicted
    ASL letter and confidence.
    """

    image = Image.open(image_path)

    image = segment_and_crop(image)

    image = test_transform(image)

    # Add batch dimension
    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.inference_mode():
        outputs = model(image)

        probabilities = torch.softmax(outputs, dim=1)

        top_probs, top_indices = torch.topk(probabilities, k=3)

    top3 = [
        (CLASSES[index.item()], probability.item() * 100)
        for probability, index in zip(top_probs[0], top_indices[0])
    ]

    predicted_letter, confidence_percentage = top3[0]

    return {
        "prediction": predicted_letter,
        "confidence": round(confidence_percentage, 2),
        "top3": top3,
    }
