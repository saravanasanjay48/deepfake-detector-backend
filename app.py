from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import requests
import io
import os
import time

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

    for attempt in range(3):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                data=image_bytes,
                timeout=30
            )
            print(f"Attempt {attempt+1} - Status: {response.status_code}")
            print(f"Response: {response.text[:300]}")

            if response.status_code == 503:
                print("Model loading, waiting 10 seconds...")
                time.sleep(10)
                continue

            if response.text.strip() == '':
                print("Empty response, waiting 5 seconds...")
                time.sleep(5)
                continue

            return response.json()

        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt+1}")
            time.sleep(5)
            continue
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            time.sleep(5)
            continue

    return {'error': 'Model unavailable after 3 attempts'}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
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
        result = query_hf_api(face_bytes)
        print(f"Final result: {result}")

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
        elif isinstance(result, dict) and 'error' in result:
            print(f"HF API error: {result}")
            return jsonify({'error': f"Model loading, please try again in 30 seconds"}), 503
        else:
            print(f"Unexpected result: {result}")
            return jsonify({'error': 'Unexpected response, please try again'}), 503

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)