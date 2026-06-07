# database.py
# Модуль для работы с базой данных (SQLite)

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DatabaseManager:
    # Класс для управления базой данных
    
    def __init__(self, db_type='sqlite', db_path='diabetes.db', 
                 host=None, database=None, user=None, password=None):

        self.db_type = db_type
        self.db_path = db_path
        self.connection = None
        self.cursor = None
        
        # Создаем директорию для БД если нужно
        if db_type == 'sqlite':
            db_dir = Path(db_path).parent
            db_dir.mkdir(exist_ok=True)
        
        # Подключаемся
        self.connect()
        
        # Инициализируем таблицы
        self.init_tables()
    
    def connect(self):
        # Подключение к базе данных
        try:
            if self.db_type == 'sqlite':
                self.connection = sqlite3.connect(self.db_path)
                self.connection.execute("PRAGMA foreign_keys = ON")
                self.connection.execute("PRAGMA journal_mode = WAL")  # Безопасный режим
                print(f"Подключено к SQLite: {self.db_path}")
            
            self.cursor = self.connection.cursor()
            return True
            
        except Exception as e:
            print(f"Ошибка подключения к БД: {str(e)}")
            return False
    
    def disconnect(self):
        # Отключение от базы данных
        if self.connection:
            self.connection.close()
            print("Отключено от базы данных")
    
    def init_tables(self):
        # Инициализация всех таблиц
        
        # Таблица пациентов (обезличенные данные)
        self._create_patients_table()
        
        # Таблица предсказаний
        self._create_predictions_table()
        
        # Таблица результатов моделей
        self._create_model_results_table()
        
        # Таблица аудита доступа
        self._create_audit_log_table()
        
        # Таблица настроек
        self._create_settings_table()
        
        print("Таблицы базы данных созданы/проверены")
    
    def _create_patients_table(self):
        # Создание таблицы пациентов
        if self.db_type == 'sqlite':
            query = """
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    patient_hash TEXT UNIQUE NOT NULL,
                    encrypted_data TEXT NOT NULL,
                    data_version INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT,
                    is_active INTEGER DEFAULT 1,
                    metadata TEXT
                )
            """
        
        self.cursor.execute(query)
        self.connection.commit()
    
    def _create_predictions_table(self):
        # Создание таблицы предсказаний
        query = """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                probability REAL,
                prediction INTEGER,
                threshold REAL,
                risk_level TEXT,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT,
                features_used TEXT,
                processing_time REAL,
                FOREIGN KEY (patient_hash) REFERENCES patients(patient_hash)
            )
        """
        self.cursor.execute(query)
        self.connection.commit()
    
    def _create_model_results_table(self):
        # Создание таблицы результатов моделей
        query = """
            CREATE TABLE IF NOT EXISTS model_results (
                result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                roc_auc REAL,
                cv_score REAL,
                training_samples INTEGER,
                test_samples INTEGER,
                parameters TEXT,
                is_current INTEGER DEFAULT 0,
                notes TEXT
            )
        """
        self.cursor.execute(query)
        self.connection.commit()
    
    def _create_audit_log_table(self):
        # Создание таблицы аудита
        query = """
            CREATE TABLE IF NOT EXISTS audit_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT,
                action TEXT,
                table_name TEXT,
                record_id TEXT,
                action_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                details TEXT,
                status TEXT
            )
        """
        self.cursor.execute(query)
        self.connection.commit()
    
    def _create_settings_table(self):
        # Создание таблицы настроек
        query = """
            CREATE TABLE IF NOT EXISTS settings (
                setting_key TEXT PRIMARY KEY,
                setting_value TEXT,
                setting_type TEXT,
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_by TEXT
            )
        """
        self.cursor.execute(query)
        self.connection.commit()
        
        # Добавляем настройки по умолчанию
        self._init_default_settings()
    
    def _init_default_settings(self):
        # Инициализация настроек по умолчанию
        default_settings = [
            ('db_version', '1.0', 'string', 'Версия базы данных'),
            ('encryption_enabled', 'True', 'boolean', 'Включено ли шифрование'),
            ('audit_enabled', 'True', 'boolean', 'Включен ли аудит'),
            ('default_model', 'Random Forest', 'string', 'Модель по умолчанию'),
            ('prediction_threshold', '0.5', 'float', 'Порог предсказания по умолчанию')
        ]
        
        for key, value, stype, desc in default_settings:
            self.cursor.execute("""
                INSERT OR IGNORE INTO settings (setting_key, setting_value, setting_type, description)
                VALUES (?, ?, ?, ?)
            """, (key, value, stype, desc))
        
        self.connection.commit()
    
    def save_patient_data(self, df, patient_id=None, encrypt_func=None):
        
        # Сохранение данных пациента в БД
        # Генерируем хеш пациента
        if patient_id is None:
            patient_id = hashlib.sha256(
                f"{datetime.now().timestamp()}{np.random.random()}".encode()
            ).hexdigest()[:16]
        
        # Создаем хеш для поиска
        patient_hash = hashlib.sha256(
            f"{patient_id}{df.to_string()}".encode()
        ).hexdigest()[:32]
        
        # Подготовка данных для сохранения
        data_to_save = df.to_json(date_format='iso', orient='records')
        
        # Шифрование если нужно
        if encrypt_func:
            data_to_save = encrypt_func(data_to_save)
        
        # Метаданные
        metadata = json.dumps({
            'columns': df.columns.tolist(),
            'shape': df.shape,
            'dtypes': df.dtypes.astype(str).to_dict(),
            'has_outcome': 'Outcome' in df.columns
        })
        
        # Сохраняем
        if self.db_type == 'sqlite':
            query = """
                INSERT OR REPLACE INTO patients 
                (patient_id, patient_hash, encrypted_data, updated_at, metadata, created_by)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?)
            """
            self.cursor.execute(query, (
                patient_id, patient_hash, data_to_save, metadata, 
                os.getenv('USER', 'unknown')
            ))
        else:
            query = """
                INSERT INTO patients 
                (patient_id, patient_hash, encrypted_data, updated_at, metadata, created_by)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP, %s::jsonb, %s)
                ON CONFLICT (patient_id) DO UPDATE SET
                encrypted_data = EXCLUDED.encrypted_data,
                updated_at = CURRENT_TIMESTAMP,
                metadata = EXCLUDED.metadata
            """
            self.cursor.execute(query, (
                patient_id, patient_hash, data_to_save, metadata,
                os.getenv('USER', 'unknown')
            ))
        
        self.connection.commit()
        
        # Логируем действие
        self.log_audit('SAVE_PATIENT', 'patients', patient_hash, 'SUCCESS')
        
        print(f"Данные пациента сохранены. Хеш: {patient_hash}")
        return patient_hash
    
    def load_patient_data(self, patient_hash, decrypt_func=None):
        # Загрузка данных пациента из БД
        query = "SELECT encrypted_data, metadata FROM patients WHERE patient_hash = ?"
        if self.db_type == 'postgresql':
            query = query.replace('?', '%s')
        
        self.cursor.execute(query, (patient_hash,))
        result = self.cursor.fetchone()
        
        if not result:
            raise ValueError(f"Пациент с хешем {patient_hash} не найден")
        
        if self.db_type == 'sqlite':
            encrypted_data, metadata = result
        else:
            encrypted_data, metadata = result
        
        # Расшифровка
        if decrypt_func:
            encrypted_data = decrypt_func(encrypted_data)
        
        # Загрузка в DataFrame
        df = pd.read_json(encrypted_data)
        
        # Логируем
        self.log_audit('LOAD_PATIENT', 'patients', patient_hash, 'SUCCESS')
        
        print(f"Данные пациента загружены. Хеш: {patient_hash}")
        return df
    
    def save_prediction(self, patient_hash, model_name, probability, prediction,
                       threshold=0.5, risk_level=None, model_version='1.0',
                       features_used=None, processing_time=None):
        
        # Сохранение результата предсказания
        if risk_level is None:
            if probability < 0.3:
                risk_level = 'Низкий'
            elif probability < 0.6:
                risk_level = 'Средний'
            else:
                risk_level = 'Высокий'
        
        features_used_json = json.dumps(features_used) if features_used else None
        
        query = """
            INSERT INTO predictions 
            (patient_hash, model_name, probability, prediction, threshold, 
             risk_level, model_version, features_used, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        if self.db_type == 'postgresql':
            query = query.replace('?', '%s')
        
        self.cursor.execute(query, (
            patient_hash, model_name, probability, int(prediction), threshold,
            risk_level, model_version, features_used_json, processing_time
        ))
        self.connection.commit()
        
        self.log_audit('SAVE_PREDICTION', 'predictions', patient_hash, 'SUCCESS')
        print(f"Предсказание сохранено для пациента {patient_hash}")
    
    def save_model_results(self, model_name, metrics, parameters=None, notes=None):
        
        # Сохранение результатов обучения модели
        # Сбрасываем флаг is_current для этой модели
        self.cursor.execute("""
            UPDATE model_results SET is_current = 0 WHERE model_name = ?
        """, (model_name,))
        
        parameters_json = json.dumps(parameters) if parameters else None
        
        query = """
            INSERT INTO model_results 
            (model_name, accuracy, precision, recall, f1_score, roc_auc, 
             cv_score, training_samples, test_samples, parameters, notes, is_current)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """
        
        self.cursor.execute(query, (
            model_name,
            metrics.get('accuracy'),
            metrics.get('precision'),
            metrics.get('recall'),
            metrics.get('f1_score'),
            metrics.get('roc_auc'),
            metrics.get('cv_score'),
            metrics.get('training_samples'),
            metrics.get('test_samples'),
            parameters_json,
            notes
        ))
        self.connection.commit()
        
        self.log_audit('SAVE_MODEL_RESULTS', 'model_results', model_name, 'SUCCESS')
        print(f"Результаты модели '{model_name}' сохранены")
    
    def get_model_history(self, model_name=None):
        # Получение истории обучения моделей
        if model_name:
            query = "SELECT * FROM model_results WHERE model_name = ? ORDER BY training_date DESC"
            self.cursor.execute(query, (model_name,))
        else:
            query = "SELECT * FROM model_results ORDER BY training_date DESC"
            self.cursor.execute(query)
        
        results = self.cursor.fetchall()
        
        if results:
            columns = [desc[0] for desc in self.cursor.description]
            df = pd.DataFrame(results, columns=columns)
            return df
        return pd.DataFrame()
    
    def get_patient_predictions(self, patient_hash, limit=10):
        # Получение истории предсказаний для пациента
        query = """
            SELECT * FROM predictions 
            WHERE patient_hash = ? 
            ORDER BY prediction_date DESC 
            LIMIT ?
        """
        
        self.cursor.execute(query, (patient_hash, limit))
        results = self.cursor.fetchall()
        
        if results:
            columns = [desc[0] for desc in self.cursor.description]
            df = pd.DataFrame(results, columns=columns)
            return df
        return pd.DataFrame()
    
    def get_statistics(self):
        # Получение статистики по базе данных
        stats = {}
        
        # Количество пациентов
        self.cursor.execute("SELECT COUNT(*) FROM patients WHERE is_active = 1")
        stats['total_patients'] = self.cursor.fetchone()[0]
        
        # Количество предсказаний
        self.cursor.execute("SELECT COUNT(*) FROM predictions")
        stats['total_predictions'] = self.cursor.fetchone()[0]
        
        # Количество обученных моделей
        self.cursor.execute("SELECT COUNT(DISTINCT model_name) FROM model_results")
        stats['total_models'] = self.cursor.fetchone()[0]
        
        # Последнее предсказание
        self.cursor.execute("SELECT MAX(prediction_date) FROM predictions")
        last_pred = self.cursor.fetchone()[0]
        stats['last_prediction_date'] = last_pred if last_pred else None
        
        # Распределение уровней риска
        self.cursor.execute("""
            SELECT risk_level, COUNT(*) FROM predictions 
            GROUP BY risk_level
        """)
        stats['risk_distribution'] = dict(self.cursor.fetchall())
        
        return stats
    
    def log_audit(self, action, table_name, record_id, status, details=None):
        # Логирование действий пользователя
        import socket
        
        query = """
            INSERT INTO audit_log 
            (user_name, action, table_name, record_id, ip_address, details, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        if self.db_type == 'postgresql':
            query = query.replace('?', '%s')
        
        try:
            user_name = os.getenv('USER', os.getenv('USERNAME', 'unknown'))
            hostname = socket.gethostname()
            
            self.cursor.execute(query, (
                user_name, action, table_name, record_id, hostname, details, status
            ))
            self.connection.commit()
        except Exception as e:
            print(f"Ошибка логирования: {str(e)}")
    
    def export_to_csv(self, table_name, output_path):
        """
        Экспорт таблицы в CSV
        
        Parameters:
        -----------
        table_name : str
            Название таблицы
        output_path : str
            Путь для сохранения
        """
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, self.connection)
        df.to_csv(output_path, index=False)
        print(f"Таблица {table_name} экспортирована в {output_path}")
    
    def backup_database(self, backup_path=None):
        # Создание резервной копии базы данных
        if backup_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"backup_{self.db_type}_{timestamp}.db"
        
        if self.db_type == 'sqlite':
            # Для SQLite просто копируем файл
            import shutil
            shutil.copy2(self.db_path, backup_path)
            print(f"Бэкап SQLite создан: {backup_path}")
        else:
            # Для PostgreSQL используем pg_dump
            import subprocess
            cmd = f"pg_dump -h {self.pg_params['host']} -U {self.pg_params['user']} -d {self.pg_params['database']} > {backup_path}"
            subprocess.run(cmd, shell=True)
            print(f"Бэкап PostgreSQL создан: {backup_path}")
    
    def cleanup_old_records(self, days=365):
        # Очистка старых записей
        import datetime
        
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
        
        # Очищаем старые предсказания
        self.cursor.execute("""
            DELETE FROM predictions 
            WHERE prediction_date < ?
        """, (cutoff_date,))
        
        # Очищаем старые логи
        self.cursor.execute("""
            DELETE FROM audit_log 
            WHERE action_time < ?
        """, (cutoff_date,))
        
        self.connection.commit()
        print(f"✅ Удалены записи старше {days} дней")
    
    def __enter__(self):
        """Контекстный менеджер"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Выход из контекстного менеджера"""
        self.disconnect()


# Функция для тестирования
def test_database():
    # Тестовая функция для проверки работы с БД
    print("="*60)
    print("ТЕСТИРОВАНИЕ МОДУЛЯ БАЗЫ ДАННЫХ")
    print("="*60)
    
    # Создаем тестовую БД
    db = DatabaseManager(db_type='sqlite', db_path='test_diabetes.db')
    
    # Создаем тестовые данные
    test_data = pd.DataFrame({
        'Pregnancies': [1, 2, 3],
        'Glucose': [85, 120, 150],
        'BloodPressure': [70, 80, 90],
        'SkinThickness': [20, 25, 30],
        'Insulin': [79, 100, 120],
        'BMI': [25.5, 28.3, 32.1],
        'DiabetesPedigreeFunction': [0.5, 0.6, 0.7],
        'Age': [30, 40, 50],
        'Outcome': [0, 0, 1]
    })
    
    # Сохраняем данные
    patient_hash = db.save_patient_data(test_data)
    print(f"\nСохранен пациент с хешем: {patient_hash}")
    
    # Загружаем данные
    loaded_data = db.load_patient_data(patient_hash)
    print(f"Загружены данные: {loaded_data.shape}")
    
    # Сохраняем предсказание
    db.save_prediction(
        patient_hash=patient_hash,
        model_name='Random Forest',
        probability=0.75,
        prediction=1,
        threshold=0.5,
        risk_level='Средний'
    )
    print(f"Предсказание сохранено")
    
    # Сохраняем результаты модели
    metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.80,
        'f1_score': 0.81,
        'roc_auc': 0.88,
        'cv_score': 0.86,
        'training_samples': 400,
        'test_samples': 100
    }
    db.save_model_results('Random Forest', metrics)
    print(f"Результаты модели сохранены")
    
    # Получаем статистику
    stats = db.get_statistics()
    print(f"\nСтатистика БД:")
    for key, value in stats.items():
        print(f"  • {key}: {value}")
    
    # Получаем историю
    history = db.get_model_history('Random Forest')
    print(f"\nИстория модели:")
    print(history[['model_name', 'accuracy', 'roc_auc', 'training_date']].to_string(index=False))
    
    # Экспорт
    db.export_to_csv('patients', 'export_patients.csv')
    
    # Закрываем соединение
    db.disconnect()
    
    print("\nТестирование базы данных завершено!")


if __name__ == "__main__":
    test_database()