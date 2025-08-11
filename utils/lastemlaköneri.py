import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import manhattan_distances

def clean_data(df):
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
            df[col] = df[col].astype(bool).astype(int)
    
    return df

def safe_bool_convert(series):
    try:
        true_values = ['true', 't', 'yes', 'y', '1', 1, True]
        false_values = ['false', 'f', 'no', 'n', '0', 0, False]
        return series.apply(
            lambda x: 1 if str(x).lower() in true_values else (0 if str(x).lower() in false_values else pd.NA)
        ).fillna(0).astype(int)
    except:
        return series.astype(bool).astype(int)

# Veri yükleme ve temizleme
df = pd.read_csv('istanbulöneri.csv')
df = clean_data(df.copy())

# Boolean sütun dönüşümü
bool_cols = ['Furnished', 'Air conditioning', 'has_balcony']
for col in bool_cols:
    if col in df.columns:
        df[col] = safe_bool_convert(df[col])

# Pipeline tanımları
preprocessor = ColumnTransformer([
    ('num', RobustScaler(), ['sqrt_m2', 'totalrooms', 'bathroom_count']),
    ('bool', 'passthrough', ['Furnished', 'Air conditioning', 'has_balcony'])
])

def hybrid_metric(x, y):
    # Özellik ağırlıkları
    weights = {
        'size': 0.3, 
        'price': 0.4,
        'location': 0.2,
        'luxury': 0.1
    }
    
    size_diff = abs(x[0] - y[0]) * weights['size']
    price_diff = abs(x[3] - y[3]) * weights['price']
    loc_diff = abs(x[4] - y[4]) * weights['location']
    lux_diff = abs(x[5] - y[5]) * weights['luxury']
    
    return size_diff + price_diff + loc_diff + lux_diff
model = Pipeline([
    ('preprocessor', preprocessor),
    ('nn', NearestNeighbors(n_neighbors=10, metric=hybrid_metric, algorithm='brute'))
])

features = ['sqrt_m2', 'totalrooms', 'bathroom_count', 
           'Furnished', 'Air conditioning', 'has_balcony']

def hybrid_recommend(property_id, df, model, weight_location=0.3, weight_luxury=0.2, max_price_diff=0.3):
    try:
        # Önce property'nin var olup olmadığını kontrol et
        if property_id not in df.index:
            print(f"{property_id} ID'li property bulunamadı")
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
            print(f"Filtreleme hatası: {str(e)}")
            filtered_df = df.copy()  # Filtreleme başarısız olursa tüm veriyi kullan
            
        # Eğer filtrelenmiş veri çok küçükse
        if len(filtered_df) < 5:
            print(f"Yeterli örnek yok ({len(filtered_df)}), tüm veri kullanılıyor")
            filtered_df = df.copy()
        
        # Modeli yeniden eğit
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
                print("Geçerli indeks bulunamadı")
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
        print(f"Beklenmeyen hata: {str(e)}")
        return pd.DataFrame()

# Model eğitimi
model.fit(df[features])

# Örnek kullanım
if not df.empty:
    try:
        sample_id = df.sample(1).index[0]
        print(f"Örnek property ID: {sample_id}")
        recommendations = hybrid_recommend(sample_id, df, model)
        
        if not recommendations.empty:
            print(recommendations[['sqrt_m2', 'totalrooms', 'bathroom_count', 
                                 'price_per_m2', 'location_score', 
                                 'luxury_score_final', 'hybrid_score']])
        else:
            print("Öneri bulunamadı")
    except Exception as e:
        print(f"Örnek çalıştırma hatası: {str(e)}")
else:
    print("Veri çerçevesi boş")