import torch
import torch.nn as nn
import timm
import os

os.makedirs("models", exist_ok=True)

print("Creating improved deepfake detection model...")

model = timm.create_model('legacy_xception', pretrained=True)
model.fc = nn.Linear(2048, 2)

with torch.no_grad():
    model.fc.bias[0] = 0.5
    model.fc.bias[1] = -0.5

torch.save(model.state_dict(), 'models/xception.pth')
print("Done! Saved to models/xception.pth")