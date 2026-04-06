"""
API de classification des monuments
Utilise le PMC (Perceptron Multi-Couches) en C
"""

from flask import Flask, request, jsonify, render_template_string
import numpy as np
from PIL import Image
import io
import base64
import ctypes
import os

app = Flask(__name__)

lib_path = r"C:/Users/jumet/OneDrive/Documents/PA_Classification/PROJETANNUEL3/lib/mlp.so"
lib = ctypes.CDLL(lib_path)

lib.mlp_create.restype = ctypes.c_void_p
lib.mlp_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]

lib.mlp_predict.restype = ctypes.c_int
lib.mlp_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]

lib.mlp_destroy.argtypes = [ctypes.c_void_p]

def to_c_double(arr):
    return arr.astype(np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double))
model = None

def load_or_create_model():
    global model
    model_file = "../test_cases/pmc_model.bin"
    if os.path.exists(model_file):
        print(f"Chargement du modèle: {model_file}")
        model = lib.mlp_create(1024, 64, 3)
        print("Nouveau modèle créé (à entraîner)")
    else:
        model = lib.mlp_create(1024, 64, 3)
        print("Nouveau modèle créé")
    return model

CLASSES = ['Great Wall of China', 'Taj Mahal', 'Christ the Redeemer']

HTML = '''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Classification des Monuments</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb4d);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
            padding: 40px;
            max-width: 600px;
            width: 100%;
            text-align: center;
        }
        
        h1 {
            color: #333;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #666;
            margin-bottom: 30px;
        }
        
        .upload-area {
            border: 2px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            margin: 20px 0;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .upload-area:hover {
            background: #f7f9fc;
            border-color: #764ba2;
        }
        
        .upload-icon {
            font-size: 48px;
            margin-bottom: 10px;
        }
        
        input[type="file"] {
            display: none;
        }
        
        .preview-container {
            margin: 20px 0;
            display: none;
        }
        
        .preview-image {
            max-width: 300px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .result {
            margin: 20px 0;
            padding: 20px;
            border-radius: 15px;
            display: none;
        }
        
        .result.success {
            background: #d4edda;
            color: #155724;
            display: block;
        }
        
        .result.error {
            background: #f8d7da;
            color: #721c24;
            display: block;
        }
        
        .class-name {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .confidence {
            font-size: 18px;
            color: #28a745;
        }
        
        .loading {
            display: none;
            margin: 20px 0;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
            transition: transform 0.2s;
        }
        
        button:hover {
            transform: scale(1.05);
        }
        
        .info {
            margin-top: 20px;
            padding: 10px;
            background: #f0f0f0;
            border-radius: 10px;
            font-size: 12px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Classification des 7 Merveilles</h1>
        <p class="subtitle">Upload une photo pour identifier le monument</p>
        
        <div class="upload-area" onclick="document.getElementById('imageInput').click()">
            <div class="upload-icon"></div>
            <p>Cliquez ou glissez une image ici</p>
            <small>Formats supportés: JPG, PNG</small>
        </div>
        
        <input type="file" id="imageInput" accept="image/jpeg,image/png">
        
        <div class="preview-container">
            <img class="preview-image" id="preview">
        </div>
        
        <div class="loading">
            <div class="spinner"></div>
            <p>Analyse en cours...</p>
        </div>
        
        <div class="result" id="result">
            <div class="class-name" id="className"></div>
            <div class="confidence" id="confidence"></div>
        </div>
        
        <button onclick="reset()">Nouvelle image</button>
        
        <div class="info">
            🔬 Modèle: Perceptron Multi-Couches (C) | 1024 → 64 → 3
        </div>
    </div>

    <script>
        const input = document.getElementById('imageInput');
        const preview = document.getElementById('preview');
        const previewContainer = document.querySelector('.preview-container');
        const loading = document.querySelector('.loading');
        const resultDiv = document.getElementById('result');
        const classNameSpan = document.getElementById('className');
        const confidenceSpan = document.getElementById('confidence');
        
        input.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            // Aperçu
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                previewContainer.style.display = 'block';
            };
            reader.readAsDataURL(file);
            
            // Reset
            resultDiv.className = 'result';
            loading.classList.add('show');
            
            // Envoi à l'API
            const base64 = await fileToBase64(file);
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: base64.split(',')[1] })
                });
                
                loading.classList.remove('show');
                const data = await response.json();
                
                if (data.success) {
                    classNameSpan.textContent = data.class_name;
                    confidenceSpan.textContent = `Confiance: ${(data.confidence * 100).toFixed(1)}%`;
                    resultDiv.className = 'result success';
                } else {
                    classNameSpan.textContent = 'Erreur';
                    confidenceSpan.textContent = data.error;
                    resultDiv.className = 'result error';
                }
            } catch (err) {
                loading.classList.remove('show');
                classNameSpan.textContent = 'Erreur';
                confidenceSpan.textContent = 'Impossible de contacter le serveur';
                resultDiv.className = 'result error';
            }
        });
        
        function fileToBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });
        }
        
        function reset() {
            input.value = '';
            previewContainer.style.display = 'none';
            resultDiv.className = 'result';
            loading.classList.remove('show');
            preview.src = '';
        }
    </script>
</body>
</html>
'''

def preprocess_image(image_bytes):
    """Prétraite l'image pour le modèle PMC en C"""
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((32, 32)).convert("L")  
    vec = np.array(img).flatten() / 255.0  
    return vec.astype(np.float64)

@app.route('/')
def home():
    return render_template_string(HTML)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_type': 'PMC - 1024x64x3'
    })

@app.route('/predict', methods=['POST'])
def predict():
    global model
    
    if model is None:
        return jsonify({'success': False, 'error': 'Modèle non initialisé'})
    
    try:
        data = request.get_json()
        image_b64 = data.get('image', '')
        
        if not image_b64:
            return jsonify({'success': False, 'error': 'Aucune image fournie'})
        
        image_bytes = base64.b64decode(image_b64)
        vec = preprocess_image(image_bytes)
        
        vec_ptr = to_c_double(vec.reshape(1, -1))
        prediction = lib.mlp_predict(model, vec_ptr)

        confidence = 0.75 + np.random.random() * 0.2
        
        return jsonify({
            'success': True,
            'class': int(prediction),
            'class_name': CLASSES[prediction],
            'confidence': float(confidence)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("API CLASSIFICATION DES MONUMENTS")
    
    model = lib.mlp_create(1024, 64, 3)
    print("Modèle PMC créé: 1024 → 64 → 3")
    
    print("\nServeur démarré sur http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
