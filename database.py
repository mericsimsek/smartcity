# database_pg.py - PostgreSQL için Veritabanı bağlantı ve işlemleri

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

conn = psycopg2.connect(
    dbname="smartcity",
    user="postgres",
    password="123asd",
    host="localhost",
    port="5432"
)
cur = conn.cursor()



# Örnek veri üretme fonksiyonu
def generate_property():
    cities = ['Istanbul', 'Ankara', 'Izmir']
    neighborhoods = {
        'Istanbul': ['Kadikoy', 'Besiktas', 'Sisli', 'Umraniye', 'Atakoy','Maltepe', 'Beylikduzu', 'Sancaktepe', 'Pendik', 'Kartal'],
        
    }
    
    city = random.choice(cities)
    neighborhood = random.choice(neighborhoods[city])
    
    return {
        'city': city,
        'neighborhood': neighborhood,
        'building_age': random.randint(1, 30),
        'totalrooms': random.randint(1, 5),
        'bathroom_count': random.randint(1, 3),
        'luxury_score': round(random.uniform(5.0, 9.9), 1),
        'furnished': random.choice([True, False]),
        'air_conditioning': random.choice([True, False]),
        'has_balcony': random.choice([True, False]),
        'price_per_m2': random.randint(5000, 25000),
        'room_density': round(random.uniform(0.1, 0.4), 2),
        'location_tier': random.randint(1, 4),
        'm2_location_interact': round(random.uniform(10, 50), 6)
    }

# 100 adet rastgele ilan ekleme
for _ in range(100):
    property_data = generate_property()
    cur.execute("""
        INSERT INTO property_data (
            city, neighborhood, building_age, totalrooms, bathroom_count, luxury_score,
            furnished, air_conditioning, has_balcony,
            price_per_m2, room_density, location_tier, m2_location_interact
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        property_data['city'],
        property_data['neighborhood'],
        property_data['building_age'],
        property_data['totalrooms'],
        property_data['bathroom_count'],
        property_data['luxury_score'],
        property_data['furnished'],
        property_data['air_conditioning'],
        property_data['has_balcony'],
        property_data['price_per_m2'],
        property_data['room_density'],
        property_data['location_tier'],
        property_data['m2_location_interact']
    ))

conn.commit()
cur.close()
conn.close()
