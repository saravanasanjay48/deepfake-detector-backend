from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification, pipeline
from facenet_pytorch import MTCNN
import torch
import io

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Face detector
mtcnn = MTCNN(keep_all=False, margin=20, device=device)

# Model 1 — SigLIP based (94% accuracy)
print("Loading Model 1...")
processor1 = AutoImageProcessor.from_pretrained("prithivMLmods/deepfake-detector-model-v1")
model1 = SiglipForImageClassification.from_pretrained("prithivMLmods/deepfake-detector-model-v1")
model1.eval()
print("Model 1 loaded!")

# Model 2 — ViT based (92% accuracy)
print("Loading Model 2...")
detector2 = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-v2-Model")
print("Model 2 loaded!")

print("All models ready!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')

    # Detect face
    face = mtcnn(image)
    if face is None:
        face_image = image.resize((512, 512))
    else:
        from torchvision import transforms
        face_image = transforms.ToPILImage()(face.cpu())

    # --- Model 1 prediction ---
    inputs1 = processor1(images=face_image, return_tensors="pt")
    with torch.no_grad():
        outputs1 = model1(**inputs1)
        probs1 = torch.nn.functional.softmax(outputs1.logits, dim=1).squeeze().tolist()
    # label 0 = fake, label 1 = real
    m1_fake = float(probs1[0])
    m1_real = float(probs1[1])

    # --- Model 2 prediction ---
    result2 = detector2(face_image)
    m2_fake = 0.5
    m2_real = 0.5
    for r in result2:
        lbl = r['label'].lower()
        score = float(r['score'])
        if 'fake' in lbl or 'deepfake' in lbl:
            m2_fake = score
        elif 'real' in lbl or 'realism' in lbl:
            m2_real = score

    # --- Combine both models ---
    fake_prob = (m1_fake * 0.55) + (m2_fake * 0.45)
    real_prob = (m1_real * 0.55) + (m2_real * 0.45)

    # Normalize
    total = fake_prob + real_prob
    fake_prob = fake_prob / total
    real_prob = real_prob / total

    label = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = max(fake_prob, real_prob) * 100

    return jsonify({
        'label': label,
        'confidence': round(float(confidence), 2),
        'fake_probability': round(float(fake_prob * 100), 2),
        'real_probability': round(float(real_prob * 100), 2),
        'analysis': {
            'deep_learning': round(float(m1_fake * 100), 2),
            'frequency': round(float(m2_fake * 100), 2),
            'noise': 0,
            'ela': 0,
            'texture': 0,
            'symmetry': 0
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)