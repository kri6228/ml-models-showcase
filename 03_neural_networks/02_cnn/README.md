# 🖼️ CNN — CIFAR-10 Image Classifier (PyTorch + GPU)

## What This Model Does
Classifies 32×32 color images into 10 categories using a
Convolutional Neural Network trained on GPU (NVIDIA RTX 2050).

## Model Performance
| Metric | Value |
|--------|-------|
| Test Accuracy | ~78-82% |
| Framework | PyTorch |
| Device | NVIDIA RTX 2050 |
| Dataset | CIFAR-10 (60,000 images) |
| Image size | 32×32×3 (RGB) |
| Classes | 10 |
| Training time | ~5-8 mins (GPU) |

## Classes
| Label | Class |
|-------|-------|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

## Architecture
```
Input (3×32×32)
→ Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.25)
→ Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.25)
→ Conv2D(128) → BN → ReLU → MaxPool → Dropout(0.25)
→ Flatten
→ Linear(256) → ReLU → Dropout(0.5)
→ Linear(10) → Output
```

## Files
| File | Description |
|------|-------------|
| `02_cnn.ipynb` | Full training notebook (PyTorch + GPU) |
| `cnn_model.pth` | Best saved model weights |

## Requirements
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
```

## Quickstart

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Define same architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = CNN().to(device)
model.load_state_dict(torch.load('cnn_model.pth',
                                  map_location=device))
model.eval()

# Preprocess your image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# Load and predict
img    = Image.open('your_image.jpg').convert('RGB')
tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(tensor)
    probs  = torch.softmax(output, dim=1)
    pred   = probs.argmax(1).item()

print(f"Predicted : {class_names[pred]}")
print(f"Confidence: {probs[0][pred]*100:.1f}%")
```

## Input
| Detail | Value |
|--------|-------|
| Input shape | (1, 3, 32, 32) |
| Pixel range | 0.0 to 1.0 (normalized) |
| Color channels | RGB |

> Any image will be automatically resized to 32×32 by the transform pipeline.

## Output
Single predicted class name from the 10 CIFAR-10 categories with confidence %.

## Key Concepts Used
| Concept | Purpose |
|---------|---------|
| Conv2D | Detect edges, shapes, textures |
| BatchNorm | Stabilize and speed up training |
| MaxPooling | Reduce spatial size, keep features |
| Dropout | Prevent overfitting |
| ReLU | Non-linear activation |
| ReduceLROnPlateau | Reduce learning rate when stuck |
| torch.save / load_state_dict | Save and reload best weights |


## 🔗 Previous Model
👈 [10 — ANN](../01_ann/)

## 🔗 Next
👉 [12 — Model Comparison](../../04_model_comparison/)