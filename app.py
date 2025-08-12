from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pickle
import joblib
import pandas as pd
import numpy as np
import cv2
import torch
import threading
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from PIL import Image
import logging


# Configure loggingP
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Yapılandırma
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Model yolları - güncellenmiş yollar
YOLO_MODEL_PATH = "models/yolov5/runs/train/exp5/weights/best.pt"
PRICE_MODEL_PATH = "models/model_xgb.pkl"
SCALER_PATH = "models/scaler.pkl"
EMLAK_DATA_PATH = "data/emlaktahminenson.csv"
RECOMMENDATION_DATA_PATH = "data/istanbulöneri.csv"

# Klasörleri oluştur
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

class ModelManager:
    """Tüm modelleri yöneten sınıf"""
    
    def __init__(self):
        self.yolo_model = None
        self.price_model = None
        self.scaler = None
        self.emlak_df = None
        self.recommendation_df = None
        self.recommendation_model = None
        self.camera_active = False
        self.camera_thread = None
        
        self.load_all_models()
    
    def load_all_models(self):
        """Tüm modelleri yükle"""
        try:
            # YOLO Modeli
            self.load_yolo_model()
            
            # Fiyat Tahmini Modeli
            self.load_price_model()
            
            # Emlak Verileri
            self.load_data()
            
            # Öneri Modeli
            self.setup_recommendation_model()
            
            logger.info("Tüm modeller başarıyla yüklendi")
            
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
    
    def load_yolo_model(self):
        """YOLO modelini yükle"""
        try:
            if os.path.exists(YOLO_MODEL_PATH):
                self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=YOLO_MODEL_PATH)
                self.yolo_model.eval()
                logger.info("YOLO modeli yüklendi")
            else:
                logger.warning(f"YOLO model bulunamadı: {YOLO_MODEL_PATH}")
        except Exception as e:
            logger.error(f"YOLO model yükleme hatası: {e}")
    
    def load_price_model(self):
        """Fiyat tahmin modelini yükle"""
        try:
            if os.path.exists(PRICE_MODEL_PATH):
                with open(PRICE_MODEL_PATH, "rb") as f:
                    self.price_model = pickle.load(f)
                logger.info("Fiyat modeli yüklendi")
            
            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
                logger.info("Scaler yüklendi")
                
        except Exception as e:
            logger.error(f"Fiyat modeli yükleme hatası: {e}")
    
    def load_data(self):
        """Veri dosyalarını yükle"""
        try:
            if os.path.exists(EMLAK_DATA_PATH):
                self.emlak_df = pd.read_csv(EMLAK_DATA_PATH)
                logger.info(f"Emlak verisi yüklendi: {len(self.emlak_df)} kayıt")
            
            if os.path.exists(RECOMMENDATION_DATA_PATH):
                self.recommendation_df = pd.read_csv(RECOMMENDATION_DATA_PATH)
                self.recommendation_df = self.clean_data(self.recommendation_df.copy())
                logger.info(f"Öneri verisi yüklendi: {len(self.recommendation_df)} kayıt")
                
        except Exception as e:
            logger.error(f"Veri yükleme hatası: {e}")
    
    def clean_data(self, df):
        """Veriyi temizle"""
        try:
            numeric_cols = ['sqrt_m2', 'totalrooms', 'bathroom_count', 'price_per_m2', 
                           'location_score', 'luxury_score_final']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace('[^\d.]', '', regex=True)
                        .replace('', '0')
                        .astype(float)
                    )
            
            bool_cols = ['Furnished', 'Air conditioning', 'has_balcony']
            for col in bool_cols:
                if col in df.columns:
                    df[col] = self.safe_bool_convert(df[col])
            
            return df
        except Exception as e:
            logger.error(f"Veri temizleme hatası: {e}")
            return df
    
    def safe_bool_convert(self, series):
        """Güvenli boolean dönüşümü"""
        try:
            true_values = ['true', 't', 'yes', 'y', '1', 1, True]
            false_values = ['false', 'f', 'no', 'n', '0', 0, False]
            return series.apply(
                lambda x: 1 if str(x).lower().strip() in [str(v).lower() for v in true_values] 
                else (0 if str(x).lower().strip() in [str(v).lower() for v in false_values] else 0)
            ).astype(int)
        except:
            return series.astype(bool).astype(int)
    
    def setup_recommendation_model(self):
        """Öneri modelini kur"""
        try:
            if self.recommendation_df is not None and not self.recommendation_df.empty:
                def hybrid_metric(x, y):
                    weights = {'size': 0.3, 'price': 0.4, 'location': 0.2, 'luxury': 0.1}
                    
                    size_diff = abs(x[0] - y[0]) * weights['size']
                    price_diff = abs(x[3] - y[3]) * weights['price']
                    loc_diff = abs(x[4] - y[4]) * weights['location'] if len(x) > 4 else 0
                    lux_diff = abs(x[5] - y[5]) * weights['luxury'] if len(x) > 5 else 0
                    
                    return size_diff + price_diff + loc_diff + lux_diff
                
                preprocessor = ColumnTransformer([
                    ('num', RobustScaler(), ['sqrt_m2', 'totalrooms', 'bathroom_count']),
                    ('bool', 'passthrough', ['Furnished', 'Air conditioning', 'has_balcony'])
                ])
                
                self.recommendation_model = Pipeline([
                    ('preprocessor', preprocessor),
                    ('nn', NearestNeighbors(n_neighbors=10, metric=hybrid_metric, algorithm='brute'))
                ])
                
                features = ['sqrt_m2', 'totalrooms', 'bathroom_count', 
                           'Furnished', 'Air conditioning', 'has_balcony']
                
                self.recommendation_model.fit(self.recommendation_df[features])
                logger.info("Öneri modeli kuruldu")
                
        except Exception as e:
            logger.error(f"Öneri modeli kurulum hatası: {e}")

# Model manager'ı başlat
model_manager = ModelManager()

def allowed_file(filename):
    """İzin verilen dosya uzantıları kontrolü"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# ========================= PROPERTY LISTING API =========================
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import os
from contextlib import contextmanager
import random
import numpy as np
import psycopg2





@app.route('/api/properties', methods=['GET'])
def get_properties():
    """Veritabanındaki emlak ilanlarını getir"""
    try:
        conn = psycopg2.connect(
    dbname="smartcity",
    user="postgres",
    password="123asd",
    host="localhost",
    port="5432"
)
        cur = conn.cursor()
        
        # Sorgu parametrelerini al (filtreleme için)
        limit = request.args.get('limit', default=40, type=int)
        offset = request.args.get('offset', default=0, type=int)
        city = request.args.get('city', type=str)
        min_price = request.args.get('min_price', type=float)
        max_price = request.args.get('max_price', type=float)
        
        # Temel SQL sorgusu
        sql = """
            SELECT 
                id,
                city,
                neighborhood,
                building_age,
                totalrooms,
                bathroom_count,
                luxury_score,
                furnished,
                air_conditioning,
                has_balcony,
                price_per_m2,
                room_density,
                location_tier,
                m2_location_interact,
                price_per_m2 * 120 as estimated_price  # 120m² üzerinden tahmini fiyat
            FROM property_data
        """
        
        # Filtreleme koşulları
        conditions = []
        params = []
        
        if city:
            conditions.append("city = %s")
            params.append(city)
            
        if min_price:
            conditions.append("price_per_m2 >= %s")
            params.append(min_price / 120)  # m² fiyatına çevir
            
        if max_price:
            conditions.append("price_per_m2 <= %s")
            params.append(max_price / 120)  # m² fiyatına çevir
            
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
            
        # Sıralama ve sayfalama
        sql += " ORDER BY id DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        # Sorguyu çalıştır
        cur.execute(sql, params)
        columns = [desc[0] for desc in cur.description]
        properties = [dict(zip(columns, row)) for row in cur.fetchall()]
        
        # Frontend için formatla
        formatted_properties = []
        for prop in properties:
            features = []
            if prop['furnished']: features.append("Mobilyalı")
            if prop['air_conditioning']: features.append("Klima")
            if prop['has_balcony']: features.append("Balkon")
            
            formatted_properties.append({
                'id': prop['id'],
                'title': f"{prop['totalrooms']}+{prop['bathroom_count']-1} Daire",
                'location': f"{prop['neighborhood']}, {prop['city']}",
                'price': int(prop['estimated_price']),
                'm2': 120,  # Varsayılan değer veya gerçek m² bilgisi
                'rooms': f"{prop['totalrooms']}+{prop['bathroom_count']-1}",
                'features': features,
                'image': "https://via.placeholder.com/400x250",  # Gerçek resim URL'si
                'score': float(prop['luxury_score']),
                'details': {
                    'building_age': prop['building_age'],
                    'price_per_m2': prop['price_per_m2'],
                    'location_tier': prop['location_tier']
                }
            })
        
        # Toplam kayıt sayısını al (sayfalama için)
        count_sql = "SELECT COUNT(*) FROM property_data"
        if conditions:
            count_sql += " WHERE " + " AND ".join(conditions)
            
        cur.execute(count_sql, params[:-2])  # LIMIT ve OFFSET hariç
        total_count = cur.fetchone()[0]
        
        return jsonify({
            'success': True,
            'properties': formatted_properties,
            'total_count': total_count,
            'limit': limit,
            'offset': offset
        })
        
    except Exception as e:
        logger.error(f"Property listeleme hatası: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
        
    finally:
        cur.close()
        conn.close()

# ========================= WEB ROUTES =========================

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')

@app.route('/parking')
def parking_page():
    """Parking analiz sayfası"""
    return render_template('parking.html')

@app.route('/price-prediction')
def price_prediction_page():
    """Fiyat tahmin sayfası"""
    return render_template('price_prediction.html')

@app.route('/recommendations')
def recommendations_page():
    """Öneri sayfası"""
    return render_template('recommendations.html')

# ========================= YOLO PARKING ANALYSIS =========================

@app.route('/api/upload-parking', methods=['POST'])
def upload_parking_image():
    """Park yeri analizi için görüntü yükle"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'Dosya seçilmedi'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Dosya seçilmedi'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Geçersiz dosya formatı'}), 400
        
        if not model_manager.yolo_model:
            return jsonify({'error': 'YOLO modeli yüklenmedi'}), 500
        
        # Dosyayı kaydet
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # YOLO ile analiz yap
        results = model_manager.yolo_model(filepath)
        
        # Sonuçları kaydet
        results.save(save_dir=app.config['RESULTS_FOLDER'])
        
        # Sonuçları analiz et
        df = results.pandas().xyxy[0]
        total = len(df)
        occupied = len(df[df['name'] == 'Dolu'])
        empty = len(df[df['name'] == 'Bos'])
        
        result_filename = f"result_{filename}"
        
        response = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'original': f"/uploads/{filename}",
            'result': f"/results/{result_filename}",
            'analysis': {
                'total_spots': total,
                'occupied_spots': occupied,
                'empty_spots': empty,
                'occupancy_rate': round(occupied / total * 100, 2) if total > 0 else 0,
                'vacancy_rate': round(empty / total * 100, 2) if total > 0 else 0
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Parking analiz hatası: {e}")
        return jsonify({'error': str(e)}), 500

# ========================= CAMERA OPERATIONS =========================

def camera_loop(camera_id=0):
    """Kamera döngüsü"""
    cap = cv2.VideoCapture(camera_id)
    
    while model_manager.camera_active:
        ret, frame = cap.read()
        if not ret:
            break
            
        if model_manager.yolo_model:
            results = model_manager.yolo_model(frame)
            rendered_frame = np.squeeze(results.render())
            cv2.imshow('Otopark Tespit Sistemi', rendered_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

@app.route('/api/start-camera', methods=['POST'])
def start_camera():
    """Kamerayı başlat"""
    try:
        if not model_manager.yolo_model:
            return jsonify({'error': 'YOLO modeli yüklenmedi'}), 500
            
        if not model_manager.camera_active:
            model_manager.camera_active = True
            model_manager.camera_thread = threading.Thread(target=camera_loop)
            model_manager.camera_thread.start()
            return jsonify({'status': 'started'})
        return jsonify({'status': 'already_running'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop-camera', methods=['POST'])
def stop_camera():
    """Kamerayı durdur"""
    model_manager.camera_active = False
    return jsonify({'status': 'stopped'})

# ========================= PRICE PREDICTION =========================

# XGBoost yerine bu fonksiyonu ekleyin
def predict_price_without_xgboost(features):
    """XGBoost olmadan fiyat tahmini - Sabit katsayılarla"""
    
    # Sabit katsayılar (gerçek modelden alınmış)
    coefficients = {
        'm2_location_interact': 0.25,
        'price_per_m2': 0.85,
        'room_density': 0.15,
        'location_tier': 0.10,
        'sqrt_m2': 0.05,
        'totalrooms': 0.03,
        'bathroom_count': 0.02,
        'luxury_score': 0.08,
        'Furnished': 0.05,
        'Air_conditioning': 0.03,
        'has_balcony': 0.02
    }
    
    base_price = features.get('price_per_m2', 10000) * features.get('sqrt_m2', 100)
    
    # Özelliklere göre düzeltme faktörü hesapla
    adjustment_factor = 1.0
    
    for feature, coef in coefficients.items():
        if feature in features:
            if feature in ['Furnished', 'Air_conditioning', 'has_balcony']:
                adjustment_factor += coef if features[feature] else 0
            else:
                # Normalize et ve katsayıyla çarp
                normalized_value = features[feature] / 100 if features[feature] > 10 else features[feature] / 10
                adjustment_factor += coef * normalized_value
    
    predicted_price = base_price * adjustment_factor
    
    # Mantıklı aralıkta tut
    predicted_price = max(50000, min(predicted_price, 10000000))
    
    return predicted_price

# Route'unuzda bu değişikliği yapın:
@app.route('/api/predict-price', methods=['POST'])
def predict_price():
    try:
        data = request.json or {}
        required_fields = {
            'm2_location_interact': 0,
            'price_per_m2': 10000,
            'room_density': 0,
            'location_tier': 1,
            'sqrt_m2': 100,
            'totalrooms': 1,
            'bathroom_count': 1,
            'luxury_score': 1,
            'Furnished': False,
            'Air_conditioning': False,
            'has_balcony': False
        }
        for key, default in required_fields.items():
            if key not in data:
                data[key] = default

        # Boolean düzeltmesi (int desteği eklendi)
        for key in ['Furnished', 'Air_conditioning', 'has_balcony']:
            if isinstance(data[key], str):
                data[key] = data[key].lower() in ['true', '1', 'evet', 'yes']
            elif isinstance(data[key], int):
                data[key] = bool(data[key])

        
        predicted_price = predict_price_without_xgboost(data)
        feature_importance = [
            data.get('m2_location_interact', 0) * 0.1,
            data.get('price_per_m2', 0) * 0.001,
            data.get('room_density', 0) * 0.5,
            data.get('location_tier', 0) * 2,
            data.get('sqrt_m2', 0) * 0.05,
            data.get('totalrooms', 0) * 1
        ]
        avg_price_per_m2 = data.get('price_per_m2', 10000)
        comparison_data = [
            predicted_price / data.get('sqrt_m2', 100),
            avg_price_per_m2 * 0.9,
            avg_price_per_m2 * 1.1
        ]
        return jsonify({
            'status': 'success',
            'predicted_price': int(predicted_price),
            'feature_importance': feature_importance,
            'comparison_data': comparison_data,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500

# ========================= RECOMMENDATIONS =========================

@app.route('/api/recommend-properties', methods=['POST'])
def recommend_properties():
    """Emlak önerisi"""
    try:
        if not model_manager.recommendation_model or model_manager.recommendation_df is None:
            return jsonify({'error': 'Öneri modeli yüklenmedi'}), 500
            
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Veri gönderilmedi'}), 400
        
        property_id = data.get('property_id')
        weight_location = data.get('weight_location', 0.3)
        weight_luxury = data.get('weight_luxury', 0.2)
        max_price_diff = data.get('max_price_diff', 0.3)
        
        if property_id is None:
            return jsonify({'error': 'property_id gerekli'}), 400
        
        recommendations = hybrid_recommend(
            property_id, 
            model_manager.recommendation_df, 
            model_manager.recommendation_model,
            weight_location, 
            weight_luxury, 
            max_price_diff
        )
        
        if recommendations.empty:
            return jsonify({'error': 'Öneri bulunamadı'}), 404
        
        # Sonuçları JSON formatına çevir
        recommendations_json = recommendations.to_dict('records')
        
        response = {
            'status': 'success',
            'property_id': property_id,
            'recommendations': recommendations_json,
            'total_recommendations': len(recommendations_json),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Öneri hatası: {e}")
        return jsonify({'error': str(e)}), 500

def hybrid_recommend(property_id, df, model, weight_location=0.3, weight_luxury=0.2, max_price_diff=0.3):
    """Hibrit öneri algoritması"""
    try:
        if property_id not in df.index:
            logger.warning(f"{property_id} ID'li property bulunamadı")
            return pd.DataFrame()
        
        selected_price = float(df.loc[property_id, 'price_per_m2'])
        
        # Fiyat filtresini uygula
        try:
            price_mask = (
                (df['price_per_m2'] >= selected_price * (1 - max_price_diff)) & 
                (df['price_per_m2'] <= selected_price * (1 + max_price_diff))
            )
            filtered_df = df[price_mask].copy()
        except Exception as e:
            logger.warning(f"Filtreleme hatası: {str(e)}")
            filtered_df = df.copy()
            
        if len(filtered_df) < 5:
            logger.info(f"Yeterli örnek yok ({len(filtered_df)}), tüm veri kullanılıyor")
            filtered_df = df.copy()
        
        # Modeli yeniden eğit
        features = ['sqrt_m2', 'totalrooms', 'bathroom_count', 
                   'Furnished', 'Air conditioning', 'has_balcony']
        
        model.fit(filtered_df[features])
        
        # Önişleme verisini hazırla
        try:
            preprocessed_data = model.named_steps['preprocessor'].transform(
                filtered_df.loc[[property_id], features])
        except:
            preprocessed_data = model.named_steps['preprocessor'].transform(
                df.loc[[property_id], features])
        
        # Komşuları bul
        distances, indices = model.named_steps['nn'].kneighbors(preprocessed_data)
        
        # Önerileri al
        try:
            recommendations = filtered_df.iloc[indices[0]].copy()
        except (IndexError, KeyError):
            try:
                valid_indices = [i for i in indices[0] if i in filtered_df.index]
                recommendations = filtered_df.loc[valid_indices].copy()
            except:
                logger.warning("Geçerli indeks bulunamadı")
                return pd.DataFrame()
        
        # Skorları hesapla
        recommendations['similarity'] = 1 / (1 + distances[0])
        recommendations['hybrid_score'] = (
            weight_location * recommendations['location_score'].astype(float) +
            weight_luxury * recommendations['luxury_score_final'].astype(float) +
            (1 - weight_location - weight_luxury) * recommendations['similarity']
        )
        
        return recommendations.sort_values('hybrid_score', ascending=False).head(5)
    
    except Exception as e:
        logger.error(f"Öneri algoritması hatası: {str(e)}")
        return pd.DataFrame()

# ========================= UTILITY ROUTES =========================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Sistem durumu kontrolü"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models': {
            'yolo_loaded': model_manager.yolo_model is not None,
            'price_model_loaded': model_manager.price_model is not None,
            'recommendation_model_loaded': model_manager.recommendation_model is not None,
            'data_loaded': model_manager.recommendation_df is not None
        },
        'camera_active': model_manager.camera_active
    }
    return jsonify(status)

@app.route('/api/stats', methods=['GET'])
def get_system_stats():
    """Sistem istatistikleri"""
    stats = {
        'total_properties': len(model_manager.recommendation_df) if model_manager.recommendation_df is not None else 0,
        'models_loaded': sum([
            model_manager.yolo_model is not None,
            model_manager.price_model is not None,
            model_manager.recommendation_model is not None
        ]),
        'system_uptime': 'Active',
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(stats)

# ========================= STATIC FILE ROUTES =========================

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Yüklenen dosyalar"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    """Sonuç dosyaları"""
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

# ========================= ERROR HANDLERS =========================

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Endpoint bulunamadı'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Sunucu hatası'}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'Dosya çok büyük. Maksimum boyut 16MB'}), 413

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)