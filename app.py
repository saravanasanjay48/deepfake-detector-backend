from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification
from facenet_pytorch import MTCNN
import torch
import io
import os

app = Flask(__name__)
CORS(app)

device = 'cpu'
print("Loading face detector...")
mtcnn = MTCNN(keep_all=False, margin=20, device=device)

print("Loading deepfake model...")
processor = AutoImageProcessor.from_pretrained("prithivMLmods/deepfake-detector-model-v1")
model = SiglipForImageClassification.from_pretrained("prithivMLmods/deepfake-detector-model-v1")
model.eval()
print("All models loaded!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')

    face = mtcnn(image)
    if face is None:
        face_image = image.resize((512, 512))
    else:
        from torchvision import transforms
        face_image = transforms.ToPILImage()(face.cpu())

    inputs = processor(images=face_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()

    fake_prob = round(float(probs[0]) * 100, 2)
    real_prob = round(float(probs[1]) * 100, 2)
    label = "FAKE" if probs[0] > probs[1] else "REAL"
    confidence = max(fake_prob, real_prob)

    return jsonify({
        'label': label,
        'confidence': round(float(confidence), 2),
        'fake_probability': round(float(fake_prob), 2),
        'real_probability': round(float(real_prob), 2),
        'analysis': {
            'deep_learning': round(float(fake_prob), 2),
            'frequency': 0,
            'noise': 0,
            'ela': 0,
            'texture': 0,
            'symmetry': 0
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)