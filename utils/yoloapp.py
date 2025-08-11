from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import subprocess
from datetime import datetime
import threading
import cv2
import numpy as np
from PIL import Image
import torch

app = Flask(__name__)

# Yapılandırma
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4'}

# Model yolları
MODEL_PATH = "C:/Users/meric/Desktop/otoknow7/yolov5/yolov5/runs/train/exp5/weights/best.pt"
DATA_YAML = "C:/Users/meric/Desktop/otoknow6/data.yaml"

# Klasörleri oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Model yükleme
def load_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
        model.eval()
        return model
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        return None

model = load_model()

# İzin verilen dosya uzantıları kontrolü
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Ana sayfa
@app.route('/')
def index():
    return render_template('index.html')

# Dosya yükleme ve analiz
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    if file and allowed_file(file.filename):
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Görüntüyü işle
        try:
            results = model(filepath)
            results.save(save_dir=app.config['RESULTS_FOLDER'])
            
            # Sonuçları analiz et
            df = results.pandas().xyxy[0]
            total = len(df)
            occupied = len(df[df['name'] == 'Dolu'])
            empty = len(df[df['name'] == 'Bos'])
            
            result_filename = f"result_{filename}"
            
            return jsonify({
                'status': 'success',
                'original': f"/uploads/{filename}",
                'result': f"/results/{result_filename}",
                'stats': {
                    'total': total,
                    'occupied': occupied,
                    'empty': empty,
                    'occupancy_rate': round(occupied / total * 100, 2) if total > 0 else 0
                }
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Geçersiz dosya formatı'}), 400

# Canlı kamera akışı
camera_active = False
camera_thread = None

def camera_loop(camera_id=0):
    global camera_active
    cap = cv2.VideoCapture(camera_id)
    
    while camera_active:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Model ile tespit yap
        results = model(frame)
        rendered_frame = np.squeeze(results.render())
        
        # Sonucu göster
        cv2.imshow('Otopark Tespit Sistemi', rendered_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active, camera_thread
    if not camera_active:
        camera_active = True
        camera_thread = threading.Thread(target=camera_loop)
        camera_thread.start()
        return jsonify({'status': 'started'})
    return jsonify({'status': 'already_running'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_active
    camera_active = False
    return jsonify({'status': 'stopped'})

# Statik dosyalar
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)