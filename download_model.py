import timm
import torch

print("Downloading Xception model...")
model = timm.create_model('xception', pretrained=True)
model.eval()

# Modify for 2-class output (Real vs Fake)
import torch.nn as nn
model.fc = nn.Linear(2048, 2)

# Save it
torch.save(model.state_dict(), 'models/xception.pth')
print("Done! Saved to models/xception.pth")