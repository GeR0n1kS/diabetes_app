# encryption.py
# Модуль для шифрования и защиты данных пациентов

import hashlib
import base64
import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # Исправленный импорт
from cryptography.hazmat.backends import default_backend
import secrets
import warnings
warnings.filterwarnings('ignore')

# Попытка импорта дополнительных библиотек
try:
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import serialization
    ASYMMETRIC_AVAILABLE = True
except ImportError:
    ASYMMETRIC_AVAILABLE = False
    print("Асимметричное шифрование недоступно")


class DataEncryptor:
    
    # Класс для шифрования и дешифрования данных пациентов
    # Поддерживает симметричное и асимметричное шифрование
        
    def __init__(self, password=None, key_file=None):
        
        # Инициализация шифровальщика
        self.password = password
        self.key_file = key_file
        self.fernet_key = None
        self.cipher_key = None
        self.salt = None
        
        # Загружаем или создаем ключ
        if key_file and os.path.exists(key_file):
            self.load_key(key_file)
        elif password:
            self.generate_key_from_password(password)
        else:
            self.generate_random_key()
    
    def generate_random_key(self):
        # Генерация случайного ключа шифрования
        self.fernet_key = Fernet.generate_key()
        self.cipher = Fernet(self.fernet_key)
        print("Сгенерирован случайный ключ шифрования")
        return self.fernet_key
    
    def generate_key_from_password(self, password, salt=None):
        # Генерация ключа из пароля с использованием PBKDF2
        if salt is None:
            salt = os.urandom(32)
        
        # Используем PBKDF2HMAC вместо PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self.fernet_key = key
        self.cipher = Fernet(self.fernet_key)
        self.salt = salt
        
        print("Ключ шифрования сгенерирован из пароля")
        return self.fernet_key, salt
    
    def save_key(self, file_path, encrypt_with_password=None):
        # Сохранение ключа в файл
        key_data = {
            'key': self.fernet_key.decode() if self.fernet_key else None,
            'salt': base64.b64encode(self.salt).decode() if hasattr(self, 'salt') and self.salt else None,
            'created_at': datetime.now().isoformat(),
            'algorithm': 'Fernet'
        }
        
        key_json = json.dumps(key_data, indent=2)
        
        # Если указан пароль, шифруем ключ
        if encrypt_with_password:
            temp_key = Fernet.generate_key()
            temp_cipher = Fernet(temp_key)
            encrypted = temp_cipher.encrypt(key_json.encode())
            
            # Сохраняем зашифрованный ключ
            with open(file_path, 'w') as f:
                f.write(base64.b64encode(encrypted).decode())
            print(f"Ключ сохранен (зашифрован) в {file_path}")
        else:
            with open(file_path, 'w') as f:
                f.write(key_json)
            print(f"Ключ сохранен в {file_path}")
    
    def load_key(self, file_path, password=None):
        # Загрузка ключа из файла
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Пробуем расшифровать если есть пароль
            if password:
                try:
                    encrypted = base64.b64decode(content)
                    temp_key = Fernet.generate_key()
                    temp_cipher = Fernet(temp_key)
                    key_json = temp_cipher.decrypt(encrypted).decode()
                except:
                    key_json = content
            else:
                key_json = content
            
            key_data = json.loads(key_json)
            self.fernet_key = key_data['key'].encode()
            self.cipher = Fernet(self.fernet_key)
            
            if 'salt' in key_data and key_data['salt']:
                self.salt = base64.b64decode(key_data['salt'])
            
            print(f"Ключ загружен из {file_path}")
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки ключа: {str(e)}")
            return False
    
    def encrypt_data(self, data):
        
        #Шифрование данных
        # Преобразуем данные в bytes
        if isinstance(data, pd.DataFrame):
            data_bytes = data.to_json(date_format='iso').encode()
        elif isinstance(data, (dict, list)):
            data_bytes = json.dumps(data).encode()
        elif isinstance(data, str):
            data_bytes = data.encode()
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            data_bytes = str(data).encode()
        
        # Шифруем
        encrypted = self.cipher.encrypt(data_bytes)
        
        # Возвращаем как base64 строку
        return base64.b64encode(encrypted).decode()
    
    def decrypt_data(self, encrypted_data):
        
        # Дешифрование данных
        # Декодируем base64
        encrypted_bytes = base64.b64decode(encrypted_data)
        
        # Расшифровываем
        decrypted = self.cipher.decrypt(encrypted_bytes)
        
        # Пробуем интерпретировать как JSON
        try:
            return json.loads(decrypted)
        except:
            try:
                return decrypted.decode()
            except:
                return decrypted
    
    def encrypt_dataframe(self, df, columns_to_encrypt=None):
        
        # Шифрование отдельных колонок DataFrame
        df_encrypted = df.copy()
        
        if columns_to_encrypt is None:
            columns_to_encrypt = df.columns
        
        for col in columns_to_encrypt:
            if col in df_encrypted.columns:
                df_encrypted[col] = df_encrypted[col].apply(
                    lambda x: self.encrypt_data(str(x)) if pd.notna(x) else x
                )
        
        return df_encrypted
    
    def decrypt_dataframe(self, df_encrypted, columns_to_decrypt=None):
        
        # Дешифрование колонок DataFrame
        df_decrypted = df_encrypted.copy()
        
        if columns_to_decrypt is None:
            columns_to_decrypt = df_decrypted.select_dtypes(include=['object']).columns
        
        for col in columns_to_decrypt:
            if col in df_decrypted.columns:
                df_decrypted[col] = df_decrypted[col].apply(
                    lambda x: self.decrypt_data(x) if isinstance(x, str) and x else x
                )
        
        return df_decrypted


class PatientDataProtector:
    
    # Класс для защиты данных пациентов
    # Включает анонимизацию, маскирование и шифрование
    
    def __init__(self, encryptor=None):
        
        # Инициализация защитника данных
        self.encryptor = encryptor or DataEncryptor()
        self.anon_rules = {}
    
    def anonymize_patient_id(self, df, id_column='PatientID', salt=None):
        
        # Анонимизация ID пациентов
        df_anon = df.copy()
        
        if salt is None:
            salt = str(datetime.now().timestamp())
        
        def hash_id(patient_id):
            data = f"{patient_id}{salt}".encode()
            return hashlib.sha256(data).hexdigest()[:16]
        
        if id_column in df_anon.columns:
            df_anon['AnonymizedID'] = df_anon[id_column].apply(hash_id)
            df_anon = df_anon.drop(columns=[id_column])
        else:
            df_anon['AnonymizedID'] = df_anon.apply(
                lambda row: hashlib.sha256(str(row.values).encode()).hexdigest()[:16],
                axis=1
            )
        
        print(f"Данные анонимизированы. Создано {len(df_anon)} анонимных ID")
        return df_anon
    
    def mask_sensitive_data(self, df, sensitive_columns=None, mask_char='*', reveal_chars=2):
        
        # Маскирование чувствительных данных
        if sensitive_columns is None:
            sensitive_columns = ['Name', 'Phone', 'Email', 'Address', 'SSN', 'Insurance']
        
        df_masked = df.copy()
        
        for col in sensitive_columns:
            if col in df_masked.columns:
                df_masked[col] = df_masked[col].apply(
                    lambda x: self._mask_value(str(x), mask_char, reveal_chars) if pd.notna(x) else x
                )
        
        print(f"Замаскировано {len(sensitive_columns)} чувствительных колонок")
        return df_masked
    
    def _mask_value(self, value, mask_char='*', reveal_chars=2):
        # Маскирование отдельного значения
        if len(value) <= reveal_chars:
            return mask_char * len(value)
        
        visible_part = value[-reveal_chars:]
        masked_part = mask_char * (len(value) - reveal_chars)
        return masked_part + visible_part
    
    def add_noise_to_numeric(self, df, columns=None, noise_level=0.01):
        
        # Добавление шума к числовым данным (для защиты от инференса)
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        df_noisy = df.copy()
        
        for col in columns:
            if col in df_noisy.columns:
                std = df_noisy[col].std()
                if std > 0:
                    noise = np.random.normal(0, noise_level * std, len(df_noisy))
                    df_noisy[col] = df_noisy[col] + noise
        
        print(f"Добавлен шум к {len(columns)} числовым колонкам (уровень: {noise_level})")
        return df_noisy
    
    def create_encrypted_backup(self, df, backup_path, password=None):
        
        # Создание зашифрованной резервной копии
        data_json = df.to_json(date_format='iso', orient='records')
        
        if password:
            encryptor = DataEncryptor(password=password)
            encrypted = encryptor.encrypt_data(data_json)
        else:
            encrypted = self.encryptor.encrypt_data(data_json)
        
        with open(backup_path, 'w') as f:
            f.write(encrypted)
        
        print(f"Зашифрованный бэкап создан: {backup_path}")
        
        metadata = {
            'created_at': datetime.now().isoformat(),
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'encrypted': True
        }
        
        with open(f"{backup_path}.meta", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_encrypted_backup(self, backup_path, password=None):
        
        # Загрузка из зашифрованной резервной копии
        with open(backup_path, 'r') as f:
            encrypted = f.read()
        
        if password:
            encryptor = DataEncryptor(password=password)
            decrypted = encryptor.decrypt_data(encrypted)
        else:
            decrypted = self.encryptor.decrypt_data(encrypted)
        
        if isinstance(decrypted, str):
            df = pd.read_json(decrypted, orient='records')
        else:
            df = pd.DataFrame(decrypted)
        
        print(f"Бэкап загружен из {backup_path}")
        return df


class AuditLogger:
    # Класс для логирования доступа к данным
    
    def __init__(self, log_file='audit.log'):
        self.log_file = log_file
        self.logs = []
        Path(log_file).parent.mkdir(exist_ok=True)
    
    def log_access(self, user, action, resource, status, details=None):
        # Логирование доступа
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user': user,
            'action': action,
            'resource': resource,
            'status': status,
            'details': details,
            'ip': self._get_ip()
        }
        
        self.logs.append(log_entry)
        self._write_to_file(log_entry)
    
    def _get_ip(self):
        try:
            import socket
            return socket.gethostbyname(socket.gethostname())
        except:
            return 'unknown'
    
    def _write_to_file(self, log_entry):
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_logs(self, limit=100, user=None, action=None):
        logs = self.logs[-limit:]
        
        if user:
            logs = [log for log in logs if log['user'] == user]
        
        if action:
            logs = [log for log in logs if log['action'] == action]
        
        return logs


# Функция для тестирования
def test_encryption():
    # Тестовая функция для проверки шифрования
    print("="*60)
    print("ТЕСТИРОВАНИЕ МОДУЛЯ ШИФРОВАНИЯ")
    print("="*60)
    
    print("\n1. ТЕСТ ШИФРОВАНИЯ ДАННЫХ")
    print("-" * 40)
    
    encryptor = DataEncryptor(password="my_secure_password_2024")
    
    test_data = {
        'patient_id': 12345,
        'name': 'Иван Петров',
        'glucose': 120,
        'bmi': 28.5,
        'diagnosis': 'Преддиабет'
    }
    
    encrypted = encryptor.encrypt_data(test_data)
    print(f"Исходные данные: {test_data}")
    print(f"Зашифрованные данные: {encrypted[:50]}...")
    
    decrypted = encryptor.decrypt_data(encrypted)
    print(f"Расшифрованные данные: {decrypted}")
    
    print("\n2. ТЕСТ ЗАЩИТЫ ДАННЫХ")
    print("-" * 40)
    
    df = pd.DataFrame({
        'PatientID': [1001, 1002, 1003],
        'Name': ['Иван Петров', 'Мария Сидорова', 'Алексей Иванов'],
        'Age': [45, 52, 38],
        'Glucose': [110, 145, 95],
        'Outcome': [0, 1, 0]
    })
    
    protector = PatientDataProtector(encryptor)
    df_anon = protector.anonymize_patient_id(df, id_column='PatientID')
    print(f"Анонимизация завершена")
    
    print("\nТестирование модуля шифрования завершено!")


if __name__ == "__main__":
    test_encryption()