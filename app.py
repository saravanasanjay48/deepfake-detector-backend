from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import requests
import io
import os
import base64

app = Flask(__name__)
CORS(app)

device = 'cpu'
print("Loading face detector...")
mtcnn = MTCNN(keep_all=False, margin=20, device=device)
print("Face detector ready!")

HF_API_URL = "https://api-inference.huggingface.co/models/prithivMLmods/deepfake-detector-model-v1"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def query_hf_api(image_bytes):
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    response = requests.post(HF_API_URL, headers=headers, data=image_bytes)
    return response.json()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Detect and crop face
    face = mtcnn(image)
    if face is None:
        face_image = image.resize((224, 224))
    else:
        from torchvision import transforms
        face_image = transforms.ToPILImage()(face.cpu())
        face_image = face_image.resize((224, 224))

    # Convert face to bytes
    buf = io.BytesIO()
    face_image.save(buf, format='JPEG')
    face_bytes = buf.getvalue()

    # Call HuggingFace API
    try:
        result = query_hf_api(face_bytes)

        if isinstance(result, list):
            fake_prob = 0.5
            real_prob = 0.5
            for item in result:
                lbl = item['label'].lower()
                score = float(item['score'])
                if 'fake' in lbl:
                    fake_prob = score
                elif 'real' in lbl:
                    real_prob = score

            label = "FAKE" if fake_prob > real_prob else "REAL"
            confidence = max(fake_prob, real_prob) * 100

            return jsonify({
                'label': label,
                'confidence': round(float(confidence), 2),
                'fake_probability': round(float(fake_prob * 100), 2),
                'real_probability': round(float(real_prob * 100), 2),
                'analysis': {
                    'deep_learning': round(float(fake_prob * 100), 2),
                    'frequency': 0,
                    'noise': 0,
                    'ela': 0,
                    'texture': 0,
                    'symmetry': 0
                }
            })
        else:
            return jsonify({'error': 'Model loading, please try again in 30 seconds'}), 503

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)