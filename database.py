# database_pg.py - PostgreSQL için Veritabanı bağlantı ve işlemleri

import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple
import os
from contextlib import contextmanager

class DatabaseManager:
    def __init__(self):
        self.config = {
            'dbname': os.getenv('DB_NAME', 'smartcity_db'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', 5432)
        }
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def get_connection(self):
        conn = None
        try:
            conn = psycopg2.connect(**self.config)
            yield conn
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def execute_query(self, query: str, params: Tuple = None, fetch: str = None):
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(query, params or ())
                    if fetch == 'one':
                        result = cursor.fetchone()
                    elif fetch == 'all':
                        result = cursor.fetchall()
                    else:
                        result = cursor.rowcount
                conn.commit()
                return result
        except Exception as e:
            self.logger.error(f"Query execution error: {e}")
            return None

    # PROPERTY OPERATIONS
    def insert_property(self, property_data: Dict) -> Optional[int]:
        query = """
        INSERT INTO properties (
            title, description, price, price_per_m2, m2, rooms, total_rooms,
            bathroom_count, location, district, neighborhood, location_tier,
            luxury_score, room_density, m2_location_interact, furnished,
            air_conditioning, has_balcony, has_elevator, has_parking
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        params = (
            property_data.get('title'),
            property_data.get('description'),
            property_data.get('price'),
            property_data.get('price_per_m2'),
            property_data.get('m2'),
            property_data.get('rooms'),
            property_data.get('total_rooms'),
            property_data.get('bathroom_count', 1),
            property_data.get('location'),
            property_data.get('district'),
            property_data.get('neighborhood'),
            property_data.get('location_tier', 3),
            property_data.get('luxury_score', 5),
            property_data.get('room_density'),
            property_data.get('m2_location_interact'),
            property_data.get('furnished', False),
            property_data.get('air_conditioning', True),
            property_data.get('has_balcony', True),
            property_data.get('has_elevator', False),
            property_data.get('has_parking', False)
        )
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    inserted_id = cursor.fetchone()[0]
                conn.commit()
                return inserted_id
        except Exception as e:
            self.logger.error(f"Property insert error: {e}")
            return None

    def get_properties(self, filters: Dict = None, limit: int = 50) -> List[Dict]:
        query = "SELECT * FROM properties WHERE is_active = TRUE"
        params = []
        if filters:
            if 'district' in filters:
                query += " AND district = %s"
                params.append(filters['district'])
            if 'rooms' in filters:
                query += " AND rooms = %s"
                params.append(filters['rooms'])
            if 'min_price' in filters:
                query += " AND price >= %s"
                params.append(filters['min_price'])
            if 'max_price' in filters:
                query += " AND price <= %s"
                params.append(filters['max_price'])
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        return self.execute_query(query, tuple(params), fetch='all') or []

    # Diğer metotlar (insert_parking_analysis, get_parking_statistics, insert_price_prediction, log_user_interaction,
    # log_performance_metric, get_daily_stats, get_popular_properties) da benzer şekilde düzenlenmeli.
    # Aynı yapıyı kullanıp parametreleri Postgres uyumlu şekilde tutmalısın.

# Flask API kısmı aynı kalabilir (ufak parametre değişiklikleri dışında).
