import torch
import torch.nn as nn
import timm
from torchvision import transforms
from facenet_pytorch import MTCNN
from PIL import Image
import sys

device = 'cpu'
mtcnn = MTCNN(keep_all=False, margin=20, device=device)

model = timm.create_model('legacy_xception', pretrained=False)
model.fc = nn.Linear(2048, 2)
model.load_state_dict(torch.load('models/xception.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    face = mtcnn(image)
    if face is None:
        print("No face detected!")
        return
    face_pil = transforms.ToPILImage()(face.cpu())
    input_tensor = transform(face_pil).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        fake_prob = probs[0][1].item()
        real_prob = probs[0][0].item()
    print(f"REAL: {real_prob*100:.1f}%  FAKE: {fake_prob*100:.1f}%")
    print(f"Verdict: {'FAKE' if fake_prob > 0.4 else 'REAL'}")

if len(sys.argv) > 1:
    predict(sys.argv[1])
else:
    print("Usage: python test_model.py path_to_image.jpg")