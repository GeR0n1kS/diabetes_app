# config.py
# Централизованный файл конфигурации для приложения прогнозирования диабета

import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# ============================================
# БАЗОВЫЕ ПУТИ
# ============================================

# Корневая директория проекта
BASE_DIR = Path(__file__).parent.absolute()

# Поддиректории
DATA_DIR = BASE_DIR / 'data'
LOGS_DIR = BASE_DIR / 'logs'
MODELS_DIR = BASE_DIR / 'saved_models'
BACKUP_DIR = BASE_DIR / 'backups'
ASSETS_DIR = BASE_DIR / 'assets'
REPORTS_DIR = BASE_DIR / 'reports'
CONFIG_DIR = BASE_DIR / 'config'

# Создаем необходимые директории
for dir_path in [DATA_DIR, LOGS_DIR, MODELS_DIR, BACKUP_DIR, ASSETS_DIR, REPORTS_DIR, CONFIG_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============================================
# НАСТРОЙКИ ПРИЛОЖЕНИЯ
# ============================================

APP_CONFIG = {
    'name': 'Diabetes Risk Prediction System',
    'version': '1.0.0',
    'license': 'OpenSourse',
    'environment': os.getenv('APP_ENV', 'development'),
    'debug': os.getenv('DEBUG', 'False').lower() == 'true',
    'log_level': os.getenv('LOG_LEVEL', 'INFO'),
}

# ============================================
# НАСТРОЙКИ GUI
# ============================================

GUI_CONFIG = {
    'window_title': f"Diabetes Risk Prediction System v{APP_CONFIG['version']}",
    'window_size': (1400, 850),
    'min_window_size': (1024, 768),
    'theme': 'dark',  # light, dark, system
    'font_family': 'Segoe UI',
    'font_size': 10,
    'colors': {
        'primary': '#2c3e50',
        'secondary': '#3498db',
        'success': '#27ae60',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'info': '#1abc9c',
        'light': '#ecf0f1',
        'dark': '#2c3e50',
        'background': '#f5f6fa',
        'text': '#2c3e50'
    },
    'language': os.getenv('GUI_LANGUAGE', 'ru'),  # ru, en
    'auto_refresh_interval': 30,  # seconds
    'max_table_rows': 100,
    'chart_dpi': 100,
    'chart_figsize': (10, 6)
}

# ============================================
# НАСТРОЙКИ БАЗЫ ДАННЫХ
# ============================================

DB_CONFIG = {
    # Тип базы данных: 'sqlite' или 'postgresql'
    'type': os.getenv('DB_TYPE', 'sqlite'),
    
    # SQLite настройки
    'sqlite': {
        'path': DATA_DIR / 'diabetes.db',
        'backup_path': BACKUP_DIR / 'diabetes_backup.db',
        'journal_mode': 'WAL',
        'timeout': 30,
        'check_same_thread': False
    },
    
    # PostgreSQL настройки (если используется)
    'postgresql': {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'diabetes_db'),
        'user': os.getenv('DB_USER', 'diabetes_user'),
        'password': os.getenv('DB_PASSWORD', ''),
        'pool_size': int(os.getenv('DB_POOL_SIZE', 5)),
        'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', 10))
    },
    
    # Общие настройки
    'pool_recycle': 3600,
    'echo': False,  # Логирование SQL запросов
    'backup_enabled': True,
    'backup_interval_days': 7,
    'cleanup_old_records_days': 365
}

# ============================================
# НАСТРОЙКИ БЕЗОПАСНОСТИ
# ============================================

SECURITY_CONFIG = {
    # Шифрование
    'encryption_enabled': os.getenv('ENCRYPTION_ENABLED', 'True').lower() == 'true',
    'encryption_key_file': CONFIG_DIR / 'encryption.key',
    'encryption_algorithm': 'Fernet',
    'key_derivation_iterations': 100000,
    
    # Аутентификация
    'auth_enabled': os.getenv('AUTH_ENABLED', 'False').lower() == 'true',
    'jwt_secret': os.getenv('JWT_SECRET', 'change_this_secret_in_production'),
    'jwt_expiration_hours': 24,
    'password_min_length': 8,
    'max_login_attempts': 5,
    'lockout_minutes': 30,
    
    # Аудит
    'audit_enabled': True,
    'audit_log_file': LOGS_DIR / 'audit.log',
    'audit_retention_days': 90,
    
    # Данные
    'data_anonymization': True,
    'data_masking': True,
    'allowed_ip_list': os.getenv('ALLOWED_IPS', '*').split(','),
    'session_timeout_minutes': 30
}

# ============================================
# НАСТРОЙКИ МОДЕЛЕЙ ML
# ============================================

ML_CONFIG = {
    # Общие настройки
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'n_jobs': -1,  # Использовать все ядра
    
    # Предобработка
    'handle_missing': 'median',  # median, mean, remove
    'scaling': 'standard',  # standard, minmax, robust
    'cols_with_zero': ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'],
    
    # Logistic Regression
    'logistic_regression': {
        'C': 1.0,
        'solver': 'lbfgs',
        'max_iter': 1000,
        'class_weight': 'balanced'
    },
    
    # Random Forest
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced'
    },
    
    # Gradient Boosting
    'gradient_boosting': {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8
    },
    
    # Neural Network
    'neural_network': {
        'hidden_layer_sizes': (64, 32),
        'activation': 'relu',
        'solver': 'adam',
        'max_iter': 500,
        'early_stopping': True,
        'validation_fraction': 0.1
    },
    
    # Пороги принятия решений
    'thresholds': {
        'low_risk': 0.3,
        'medium_risk': 0.6,
        'high_risk': 0.8
    },
    
    # Пути сохранения
    'models_save_path': MODELS_DIR,
    'scaler_save_path': MODELS_DIR / 'scaler.pkl',
    'imputer_save_path': MODELS_DIR / 'imputer.pkl'
}

# ============================================
# НАСТРОЙКИ ВИЗУАЛИЗАЦИИ
# ============================================

VIZ_CONFIG = {
    # Общие настройки
    'style': 'seaborn-v0_8-darkgrid',
    'palette': 'husl',
    'dpi': 100,
    'figsize': {
        'small': (8, 6),
        'medium': (10, 8),
        'large': (12, 10)
    },
    
    # Цвета
    'colors': {
        'positive': '#e74c3c',  # красный для диабета
        'negative': '#27ae60',  # зеленый для здоровых
        'neutral': '#95a5a6',   # серый для нейтрального
        'models': ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    },
    
    # Настройки шрифтов
    'font': {
        'family': 'sans-serif',
        'title_size': 14,
        'label_size': 11,
        'tick_size': 9,
        'legend_size': 9
    },
    
    # Экспорт
    'export_formats': ['png', 'svg', 'pdf'],
    'export_dpi': 150,
    'save_figures': True,
    'figures_path': REPORTS_DIR / 'figures'
}

# ============================================
# НАСТРОЙКИ ДАННЫХ
# ============================================

DATA_CONFIG = {
    # Источники данных
    'default_url': 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv',
    'data_columns': [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ],
    
    # Типы колонок
    'column_types': {
        'numerical': ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
        'categorical': [],
        'target': 'Outcome'
    },
    
    # Значения по умолчанию
    'default_values': {
        'Glucose': 100,
        'BloodPressure': 80,
        'SkinThickness': 25,
        'Insulin': 80,
        'BMI': 25.0,
        'Age': 40,
        'Pregnancies': 0,
        'DiabetesPedigreeFunction': 0.5
    },
    
    # Валидация
    'value_ranges': {
        'Pregnancies': (0, 20),
        'Glucose': (0, 300),
        'BloodPressure': (0, 200),
        'SkinThickness': (0, 100),
        'Insulin': (0, 1000),
        'BMI': (10, 60),
        'DiabetesPedigreeFunction': (0, 2.5),
        'Age': (0, 120)
    },
    
    # Кэширование
    'cache_enabled': True,
    'cache_dir': DATA_DIR / 'cache',
    'cache_ttl_hours': 24
}

# ============================================
# НАСТРОЙКИ ЛОГИРОВАНИЯ
# ============================================

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        },
        'json': {
            'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}'
        }
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': LOGS_DIR / 'app.log',
            'maxBytes': 10485760,  # 10 MB
            'backupCount': 5,
            'formatter': 'detailed'
        },
        'error_file': {
            'level': 'ERROR',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': LOGS_DIR / 'error.log',
            'maxBytes': 10485760,
            'backupCount': 5,
            'formatter': 'detailed'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file', 'error_file']
    },
    'loggers': {
        'database': {
            'level': 'INFO',
            'handlers': ['file'],
            'propagate': False
        },
        'models': {
            'level': 'INFO',
            'handlers': ['file'],
            'propagate': False
        },
        'security': {
            'level': 'WARNING',
            'handlers': ['file', 'error_file'],
            'propagate': False
        }
    }
}

# ============================================
# НАСТРОЙКИ ОТЧЕТОВ
# ============================================

REPORT_CONFIG = {
    'formats': ['pdf', 'html', 'excel', 'csv'],
    'default_format': 'html',
    'template_dir': ASSETS_DIR / 'templates',
    'include_charts': True,
    'include_recommendations': True,
    'company_logo': ASSETS_DIR / 'logo.png',
    'report_footer': f"Generated by {APP_CONFIG['name']} v{APP_CONFIG['version']}",
    'auto_generate': True,
    'save_path': REPORTS_DIR
}

# ============================================
# КЛИНИЧЕСКИЕ РЕКОМЕНДАЦИИ
# ============================================

CLINICAL_CONFIG = {
    'risk_levels': {
        'low': {
            'threshold': 0.3,
            'color': 'green',
            'recommendation_ru': 'Риск диабета низкий. Рекомендуется стандартное профилактическое обследование раз в год.',
            'recommendation_en': 'Low diabetes risk. Recommended annual standard preventive examination.',
            'actions': [
                'Здоровое питание',
                'Физическая активность 150 мин/неделю',
                'Контроль массы тела'
            ]
        },
        'medium': {
            'threshold': 0.6,
            'color': 'orange',
            'recommendation_ru': 'Выявлен средний риск диабета. Требуется коррекция образа жизни и регулярный контроль.',
            'recommendation_en': 'Medium diabetes risk detected. Lifestyle modification and regular monitoring required.',
            'actions': [
                'Диета с ограничением углеводов',
                'Ежедневная физическая активность',
                'Контроль глюкозы каждые 3 месяца',
                'Консультация эндокринолога'
            ]
        },
        'high': {
            'threshold': 0.8,
            'color': 'red',
            'recommendation_ru': 'ВЫСОКИЙ РИСК ДИАБЕТА! Необходимо срочное обследование и лечение.',
            'recommendation_en': 'HIGH DIABETES RISK! Urgent examination and treatment required.',
            'actions': [
                'Срочная консультация эндокринолога',
                'Расширенное лабораторное обследование',
                'Медикаментозная профилактика',
                'Строгая диета',
                'Ежедневный мониторинг глюкозы'
            ]
        }
    },
    
    'feature_importance_threshold': 0.05,  # Минимальная важность для включения в отчет
    
    'clinical_notes': {
        'Glucose': 'Уровень глюкозы - основной фактор риска. Целевой уровень: < 5.6 ммоль/л натощак.',
        'BMI': 'Индекс массы тела - критический фактор. Целевой диапазон: 18.5-24.9.',
        'Age': 'С возрастом риск увеличивается. Рекомендуется регулярный скрининг после 45 лет.',
        'BloodPressure': 'Артериальная гипертензия повышает риск. Целевой уровень: < 130/80 мм рт.ст.',
        'Pregnancies': 'Гестационный диабет в анамнезе повышает риск развития диабета 2 типа.'
    }
}

# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================

def get_config_summary():
    # Получение сводки конфигурации
    summary = {
        'app_name': APP_CONFIG['name'],
        'app_version': APP_CONFIG['version'],
        'environment': APP_CONFIG['environment'],
        'db_type': DB_CONFIG['type'],
        'encryption_enabled': SECURITY_CONFIG['encryption_enabled'],
        'auth_enabled': SECURITY_CONFIG['auth_enabled'],
        'audit_enabled': SECURITY_CONFIG['audit_enabled'],
        'models_count': len(ML_CONFIG.keys()),
        'data_sources': ['URL', 'CSV', 'Excel', 'Database']
    }
    return summary

def validate_config():
    # Валидация конфигурации
    errors = []
    warnings = []
    
    # Проверяем необходимые директории
    required_dirs = [DATA_DIR, LOGS_DIR, MODELS_DIR, BACKUP_DIR]
    for dir_path in required_dirs:
        if not dir_path.exists():
            errors.append(f"Директория не существует: {dir_path}")
    
    # Проверяем настройки безопасности в production
    if APP_CONFIG['environment'] == 'production':
        if SECURITY_CONFIG['encryption_enabled'] and not SECURITY_CONFIG['encryption_key_file'].exists():
            warnings.append("Ключ шифрования не найден. Создайте ключ для production.")
        
        if DB_CONFIG['type'] == 'sqlite':
            warnings.append("В production рекомендуется использовать PostgreSQL вместо SQLite")
        
        if SECURITY_CONFIG['jwt_secret'] == 'change_this_secret_in_production':
            errors.append("Измените JWT_SECRET в production окружении")
    
    # Проверяем пороговые значения
    thresholds = ML_CONFIG['thresholds']
    if not (0 < thresholds['low_risk'] < thresholds['medium_risk'] < thresholds['high_risk'] < 1):
        errors.append("Некорректные пороговые значения риска")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def save_config_to_file(filepath=None):
    # Сохранение текущей конфигурации в файл
    if filepath is None:
        filepath = CONFIG_DIR / f'config_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    import json
    
    config_dict = {
        'app': APP_CONFIG,
        'gui': GUI_CONFIG,
        'database': DB_CONFIG,
        'security': SECURITY_CONFIG,
        'ml': ML_CONFIG,
        'visualization': VIZ_CONFIG,
        'data': DATA_CONFIG,
        'clinical': CLINICAL_CONFIG
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Конфигурация сохранена в {filepath}")
    return filepath

def load_config_from_file(filepath):
    # Загрузка конфигурации из файла
    import json
    
    with open(filepath, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    # Обновляем глобальные конфиги
    globals().update({
        'APP_CONFIG': config_dict.get('app', APP_CONFIG),
        'GUI_CONFIG': config_dict.get('gui', GUI_CONFIG),
        'DB_CONFIG': config_dict.get('database', DB_CONFIG),
        'SECURITY_CONFIG': config_dict.get('security', SECURITY_CONFIG),
        'ML_CONFIG': config_dict.get('ml', ML_CONFIG),
        'VIZ_CONFIG': config_dict.get('visualization', VIZ_CONFIG),
        'DATA_CONFIG': config_dict.get('data', DATA_CONFIG),
        'CLINICAL_CONFIG': config_dict.get('clinical', CLINICAL_CONFIG)
    })
    
    print(f"Конфигурация загружена из {filepath}")

def create_env_example():
    # Создание примерного .env файла
    env_example = """
# Diabetes Risk Prediction System - Environment Variables

# Application Settings
APP_ENV=development
DEBUG=False
LOG_LEVEL=INFO

# GUI Settings
GUI_LANGUAGE=ru

# Database Settings
DB_TYPE=sqlite

# Security Settings
ENCRYPTION_ENABLED=True
AUTH_ENABLED=False
JWT_SECRET=change_this_secret_in_production
ALLOWED_IPS=*

# API Settings (optional)
API_ENABLED=False
API_HOST=127.0.0.1
API_PORT=5000
API_KEY=change_this_api_key
"""
    
    env_file = BASE_DIR / '.env.example'
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write(env_example.strip())
    
    print(f"Пример .env файла создан: {env_file}")
    return env_file

# ============================================
# ИНИЦИАЛИЗАЦИЯ ПРИ ЗАГРУЗКЕ
# ============================================

# Создаем .env.example если не существует
if not (BASE_DIR / '.env.example').exists():
    create_env_example()

# Проверяем конфигурацию
config_validation = validate_config()
if not config_validation['valid']:
    print("Ошибки в конфигурации:")
    for error in config_validation['errors']:
        print(f"  • {error}")

if config_validation['warnings']:
    print("Предупреждения конфигурации:")
    for warning in config_validation['warnings']:
        print(f"  • {warning}")

# ============================================
# ТЕСТИРОВАНИЕ
# ============================================

def test_config():
    """Тестовая функция для проверки конфигурации"""
    print("="*60)
    print("ТЕСТИРОВАНИЕ КОНФИГУРАЦИИ")
    print("="*60)
    
    # Выводим сводку
    print("\nСводка конфигурации:")
    summary = get_config_summary()
    for key, value in summary.items():
        print(f"  • {key}: {value}")
    
    # Проверяем пути
    print("\nПроверка путей:")
    paths = {
        'BASE_DIR': BASE_DIR,
        'DATA_DIR': DATA_DIR,
        'LOGS_DIR': LOGS_DIR,
        'MODELS_DIR': MODELS_DIR,
        'BACKUP_DIR': BACKUP_DIR
    }
    
    for name, path in paths.items():
        print(f"  {name}: {path}")
    
    # Проверяем настройки моделей
    print("\nНастройки ML моделей:")
    for model_name in ['logistic_regression', 'random_forest', 'gradient_boosting', 'neural_network']:
        if model_name in ML_CONFIG:
            params = ML_CONFIG[model_name]
            param_count = len(params)
            print(f"  • {model_name}: {param_count} параметров")
    
    # Проверяем клинические рекомендации
    print("\nКлинические пороги:")
    for level, config in CLINICAL_CONFIG['risk_levels'].items():
        print(f"  • {level.upper()}: < {config['threshold']} - {config['recommendation_ru'][:50]}...")
    
    # Сохраняем тестовую конфигурацию
    print("\nТестовое сохранение конфигурации...")
    saved_file = save_config_to_file()
    
    print("\nТестирование конфигурации завершено!")
    return True


if __name__ == "__main__":
    test_config()