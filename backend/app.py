import os
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)
CORS(app) 

MODEL = None
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(model_path, num_classes=2):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model başarıyla yüklendi: {model_path}")
        print(f"Epoch: {checkpoint['epoch']}, Accuracy: {checkpoint['acc']:.4f}")
    except Exception as e:
        print(f"Model yüklenirken hata oluştu: {e}")
        return None
    
    return model

def predict_image(image_data, model):
    try:
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
    except Exception as e:
        print(f"Hata: Görüntü yüklenemedi - {e}")
        return None
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)
    
    model.eval()
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    
    prediction_idx = preds.item()
    
    class_names = ['fake', 'real']
    prediction_class = class_names[prediction_idx]
    confidence = probs[0][prediction_idx].item() * 100
    
    class_probs = {class_names[i]: probs[0][i].item() * 100 for i in range(len(class_names))}
    
    img_thumb = img.copy()
    img_thumb.thumbnail((300, 300))
    
    buffered = io.BytesIO()
    img_thumb.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return {
        'prediction': prediction_class,
        'confidence': confidence,
        'class_probabilities': class_probs,
        'image': img_base64
    }

@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'Görüntü bulunamadı'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    try:
        image_data = file.read()
        
        result = predict_image(image_data, MODEL)
        
        if result:
            return jsonify(result), 200
        else:
            return jsonify({'error': 'Görüntü analiz edilemedi'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({'status': 'online', 'model_loaded': MODEL is not None}), 200

if __name__ == '__main__':
    model_path = "models/best_model_20250518_182720.pth"
    
    if not os.path.exists(model_path):
        print(f"Model dosyası bulunamadı: {model_path}")
    else:
        MODEL = load_model(model_path)
        if MODEL:
            MODEL = MODEL.to(DEVICE)
    
    app.run(debug=True, host='0.0.0.0', port=5000)