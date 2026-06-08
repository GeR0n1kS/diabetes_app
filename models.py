# =======================================================
# Модуль для обучения и оценки моделей машинного обучения
# =======================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve, 
                            confusion_matrix, classification_report,
                            precision_recall_curve)
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')


class DiabetesModels:
    # Класс для управления моделями прогнозирования диабета
    
    def __init__(self):
        # Инициализация класса моделей
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.models = {}
        self.results = []
        self.best_model = None
        self.best_model_name = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        
    def load_data(self, df):
        
        # Загрузка и предобработка данных
        try:
            self.df = df.copy()
            
            # Проверка наличия целевой переменной
            if 'Outcome' not in self.df.columns:
                raise ValueError("В данных отсутствует колонка 'Outcome'")
            
            # Определяем признаки, которые могут содержать нули (пропуски)
            self.feature_names = self.df.drop('Outcome', axis=1).columns.tolist()
            
            print(f"Данные загружены. Размер: {self.df.shape}")
            print(f"Признаки: {self.feature_names}")
            print(f"Баланс классов:\n{self.df['Outcome'].value_counts()}")
            
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки данных: {str(e)}")
            return False
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        
        # Предобработка данных: обработка пропусков, разделение, масштабирование
        print("\n" + "="*50)
        print("ПРЕДОБРАБОТКА ДАННЫХ")
        print("="*50)
        
        # Определяем колонки, где нули могут означать пропуски
        cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        
        # Заменяем нули на NaN для указанных колонок
        for col in cols_with_zero:
            if col in self.df.columns:
                zero_count = (self.df[col] == 0).sum()
                if zero_count > 0:
                    print(f"  {col}: {zero_count} нулевых значений заменены на NaN")
                    self.df[col] = self.df[col].replace(0, np.nan)
        
        # Разделяем признаки и целевую переменную
        X = self.df.drop('Outcome', axis=1)
        y = self.df['Outcome']
        
        # Заполняем пропуски медианой
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = self.imputer.fit_transform(X)
        print(f"  Пропуски заполнены медианой")
        
        # Разделяем на train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_imputed, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"  Размер train: {self.X_train.shape[0]} строк")
        print(f"  Размер test: {self.X_test.shape[0]} строк")
        
        # Масштабируем признаки
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        print(f"  Данные масштабированы (StandardScaler)")
        
        return True
    
    def init_models(self):
        # Инициализация моделей машинного обучения
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced',
                C=1.0,
                solver='lbfgs'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8
            ),
            'MLP Neural Network': MLPClassifier(
                hidden_layer_sizes=(64, 32), 
                activation='relu',
                solver='adam',
                max_iter=500, 
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                verbose=False
            )
        }
        
        print(f"\nИнициализировано {len(self.models)} моделей:")
        for name in self.models.keys():
            print(f"  • {name}")
    
    def train_all_models(self):
        
        # Обучение всех моделей и сбор метрик
        if self.X_train_scaled is None:
            raise ValueError("Данные не предобработаны. Сначала вызовите preprocess_data()")
        
        print("\n" + "="*50)
        print("ОБУЧЕНИЕ МОДЕЛЕЙ")
        print("="*50)
        
        self.results = []
        self.predictions = {}
        self.probabilities = {}
        
        for name, model in self.models.items():
            print(f"\n{name}")
            print("-" * 30)
            
            try:
                # Для Gradient Boosting используем веса классов
                if name == 'Gradient Boosting':
                    classes = np.unique(self.y_train)
                    weights = compute_class_weight('balanced', classes=classes, y=self.y_train)
                    class_weight_dict = dict(zip(classes, weights))
                    sample_weights = np.array([class_weight_dict[int(y)] for y in self.y_train])
                    model.fit(self.X_train_scaled, self.y_train, sample_weight=sample_weights)
                else:
                    model.fit(self.X_train_scaled, self.y_train)
                
                # Предсказания
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
                
                # Сохраняем вероятности для графиков
                self.predictions[name] = y_pred
                self.probabilities[name] = y_pred_proba
                
                # Вычисляем метрики
                acc = accuracy_score(self.y_test, y_pred)
                prec = precision_score(self.y_test, y_pred)
                rec = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                roc_auc = roc_auc_score(self.y_test, y_pred_proba)
                
                # Кросс-валидация (только для некоторых моделей)
                cv_score = None
                if name != 'MLP Neural Network':  # MLP обучается дольше
                    cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, cv=5, scoring='roc_auc')
                    cv_score = cv_scores.mean()
                
                self.results.append({
                    'Model': name,
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1-Score': f1,
                    'ROC-AUC': roc_auc,
                    'CV-AUC': cv_score if cv_score else roc_auc
                })
                
                print(f"  Точность:     {acc:.4f}")
                print(f"  Полнота:      {rec:.4f}")
                print(f"  F1-мера:      {f1:.4f}")
                print(f"  ROC-AUC:      {roc_auc:.4f}")
                
            except Exception as e:
                print(f"  Ошибка: {str(e)}")
        
        # Определяем лучшую модель по ROC-AUC
        results_df = pd.DataFrame(self.results)
        best_idx = results_df['ROC-AUC'].idxmax()
        self.best_model_name = results_df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        print("\n" + "="*50)
        print(f"ЛУЧШАЯ МОДЕЛЬ: {self.best_model_name}")
        print(f"   ROC-AUC: {results_df.loc[best_idx, 'ROC-AUC']:.4f}")
        print("="*50)
        
        return results_df
    
    def get_feature_importance(self):
        
        # Получение важности признаков (для моделей, поддерживающих это)
        if 'Random Forest' not in self.models:
            return None
        
        rf_model = self.models['Random Forest']
        
        if hasattr(rf_model, 'feature_importances_'):
            importances = rf_model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            return importance_df
        
        return None
    
    def get_optimal_threshold(self, model_name=None):
        
        # Нахождение оптимального порога принятия решения
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.probabilities:
            return None
        
        y_scores = self.probabilities[model_name]
        
        thresholds = np.arange(0.1, 0.9, 0.02)
        best_f1 = 0
        best_threshold = 0.5
        best_precision = 0
        best_recall = 0
        
        results = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
        
        return {
            'model': model_name,
            'optimal_threshold': best_threshold,
            'precision': best_precision,
            'recall': best_recall,
            'f1_score': best_f1,
            'all_thresholds': pd.DataFrame(results)
        }
    
    def get_confusion_matrix(self, model_name=None, threshold=None):
        
        # Получение матрицы ошибок
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.probabilities:
            return None
        
        y_scores = self.probabilities[model_name]
        
        if threshold is None:
            y_pred = self.predictions[model_name]
        else:
            y_pred = (y_scores >= threshold).astype(int)
        
        cm = confusion_matrix(self.y_test, y_pred)
        
        return {
            'tn': cm[0, 0],  # True Negatives
            'fp': cm[0, 1],  # False Positives
            'fn': cm[1, 0],  # False Negatives
            'tp': cm[1, 1],  # True Positives
            'matrix': cm
        }
    
    def predict_patient(self, patient_data, model_name=None, threshold=None):
        
        # Предсказание риска диабета для одного пациента
        if self.best_model is None:
            raise ValueError("Модели не обучены. Сначала вызовите train_all_models()")
        
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models.get(model_name)
            if model is None:
                raise ValueError(f"Модель {model_name} не найдена")
        
        # Преобразуем входные данные в массив
        if isinstance(patient_data, dict):
            # Сортируем признаки в правильном порядке
            input_data = [patient_data.get(feature, 0) for feature in self.feature_names]
        elif isinstance(patient_data, (list, np.ndarray)):
            input_data = patient_data
        else:
            raise ValueError("patient_data должен быть dict или list")
        
        # Масштабируем данные
        input_scaled = self.scaler.transform([input_data])
        
        # Получаем вероятность
        probability = model.predict_proba(input_scaled)[0, 1]
        
        # Применяем порог
        if threshold is None:
            optimal = self.get_optimal_threshold(model_name)
            threshold = optimal['optimal_threshold'] if optimal else 0.5
        
        prediction = 1 if probability >= threshold else 0
        
        # Определяем уровень риска
        if probability < 0.3:
            risk_level = "Низкий"
            risk_color = "green"
        elif probability < 0.6:
            risk_level = "Средний"
            risk_color = "orange"
        else:
            risk_level = "Высокий"
            risk_color = "red"
        
        return {
            'probability': probability,
            'prediction': prediction,
            'threshold_used': threshold,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'model_used': model_name,
            'recommendation': self._get_recommendation(probability, risk_level)
        }
    
    def _get_recommendation(self, probability, risk_level):
        # Генерация рекомендаций на основе риска
        if risk_level == "Низкий":
            return "Риск диабета низкий. Рекомендуется стандартное профилактическое обследование раз в год."
        elif risk_level == "Средний":
            return "Выявлен средний риск диабета. Рекомендуется: контроль уровня глюкозы, коррекция питания, физическая активность, повторное обследование через 3 месяца."
        else:
            return "Высокий риск диабета! НЕОБХОДИМО: срочно обратиться к эндокринологу, провести расширенное обследование, начать профилактическое лечение."
    
    def get_model_summary(self):
        
        # Получение сводной информации о моделях
        if not self.results:
            return None
        
        results_df = pd.DataFrame(self.results)
        
        summary = {
            'total_models': len(self.models),
            'best_model': self.best_model_name,
            'best_roc_auc': results_df['ROC-AUC'].max(),
            'average_accuracy': results_df['Accuracy'].mean(),
            'average_f1': results_df['F1-Score'].mean(),
            'results_df': results_df
        }
        
        return summary
    
    def save_models(self, path='saved_models/'):
        
        # Сохранение обученных моделей
        import joblib
        import os
        
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            filename = f"{path}{name.replace(' ', '_')}.pkl"
            joblib.dump(model, filename)
            print(f"Модель {name} сохранена в {filename}")
        
        # Сохраняем скейлер и импутер
        joblib.dump(self.scaler, f"{path}scaler.pkl")
        joblib.dump(self.imputer, f"{path}imputer.pkl")
        
        print(f"Скейлер и импутер сохранены")
    
    def load_models(self, path='saved_models/'):
        
        # Загрузка сохраненных моделей
        import joblib
        
        model_files = {
            'Logistic Regression': 'Logistic_Regression.pkl',
            'Random Forest': 'Random_Forest.pkl',
            'Gradient Boosting': 'Gradient_Boosting.pkl',
            'MLP Neural Network': 'MLP_Neural_Network.pkl'
        }
        
        for name, filename in model_files.items():
            try:
                self.models[name] = joblib.load(f"{path}{filename}")
                print(f"Модель {name} загружена")
            except:
                print(f"Не удалось загрузить {name}")
        
        # Загружаем скейлер и импутер
        self.scaler = joblib.load(f"{path}scaler.pkl")
        self.imputer = joblib.load(f"{path}imputer.pkl")
        
        print(f"Скейлер и импутер загружены")
    
    def get_optimal_threshold(self, model_name=None):

        # Нахождение оптимального порога принятия решения
        
        if model_name is None:
            model_name = self.best_model_name
        
        y_scores = self.probabilities[model_name]
        thresholds = np.arange(0.1, 0.9, 0.02)
        
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return {
            'model': model_name,
            'optimal_threshold': best_threshold,
            'f1_score': best_f1
        }

    def predict_patient(self, patient_data, model_name=None, threshold=None):
        # ПРЕДСКАЗАНИЕ РИСКА ДИАБЕТА ДЛЯ ОДНОГО ПАЦИЕНТА
        
        # Проверка: есть ли обученная модель
        if self.best_model is None:
            raise ValueError("Модели не обучены. Сначала вызовите train_all_models()")
        
        # Выбор модели
        if model_name is None:
            model = self.best_model
        else:
            model = self.models.get(model_name)
        
        # Масштабирование входных данных
        input_scaled = self.scaler.transform([patient_data])
        
        # Получение вероятности
        probability = model.predict_proba(input_scaled)[0, 1]
        
        # Определение порога
        if threshold is None:
            optimal = self.get_optimal_threshold()
            threshold = optimal['optimal_threshold'] if optimal else 0.5
        
        # Принятие решения
        prediction = 1 if probability >= threshold else 0
        
        # Определение уровня риска
        if probability < 0.3:
            risk_level = "Низкий"
            risk_color = "green"
        elif probability < 0.6:
            risk_level = "Средний"
            risk_color = "orange"
        else:
            risk_level = "Высокий"
            risk_color = "red"
        
        return {
            'probability': probability,
            'prediction': prediction,
            'threshold_used': threshold,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'model_used': model_name if model_name else self.best_model_name,
            'recommendation': self._get_recommendation(probability, risk_level)
        }
      

    def _get_recommendation(self, probability, risk_level):

        # Генерация текстовой рекомендации на основе уровня риска

        if risk_level == "Низкий":
            return "Риск диабета низкий. Рекомендуется стандартное профилактическое обследование раз в год."
        elif risk_level == "Средний":
            return "Выявлен средний риск диабета. Рекомендуется: контроль уровня глюкозы, коррекция питания, физическая активность, повторное обследование через 3 месяца."
        else:
            return "ВЫСОКИЙ РИСК ДИАБЕТА! Необходимо срочно обратиться к эндокринологу, провести расширенное обследование, начать профилактическое лечение."


# Функция для быстрого тестирования модуля
def test_models():
    
    # Тестовая функция для проверки модуля моделей
    print("="*60)
    print("ТЕСТИРОВАНИЕ МОДУЛЯ ПРОГНОЗИРОВАНИЯ ДИАБЕТА")
    print("="*60)
    
    # Загружаем тестовые данные
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    df = pd.read_csv(url, names=columns)
    
    # Создаем экземпляр класса
    diabetes_models = DiabetesModels()
    
    # Загружаем данные
    diabetes_models.load_data(df)
    
    # Предобработка
    diabetes_models.preprocess_data()
    
    # Инициализация моделей
    diabetes_models.init_models()
    
    # Обучение
    results = diabetes_models.train_all_models()
    
    # Вывод результатов
    print("\n" + "="*50)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*50)
    print(results.round(4).to_string(index=False))
    
    # Важность признаков
    importance = diabetes_models.get_feature_importance()
    if importance is not None:
        print("\n" + "="*50)
        print("ВАЖНОСТЬ ПРИЗНАКОВ")
        print("="*50)
        print(importance.round(4).to_string(index=False))
    
    # Оптимальный порог
    optimal = diabetes_models.get_optimal_threshold()
    if optimal:
        print("\n" + "="*50)
        print("ОПТИМАЛЬНЫЙ ПОРОГ")
        print("="*50)
        print(f"Модель: {optimal['model']}")
        print(f"Порог: {optimal['optimal_threshold']:.3f}")
        print(f"Precision: {optimal['precision']:.4f}")
        print(f"Recall: {optimal['recall']:.4f}")
        print(f"F1-score: {optimal['f1_score']:.4f}")
    
    # Пример предсказания для нового пациента
    print("\n" + "="*50)
    print("ПРИМЕР ПРЕДСКАЗАНИЯ")
    print("="*50)
    
    # Новый пациент
    new_patient = {
        'Pregnancies': 2,
        'Glucose': 120,
        'BloodPressure': 70,
        'SkinThickness': 25,
        'Insulin': 80,
        'BMI': 28.5,
        'DiabetesPedigreeFunction': 0.5,
        'Age': 45
    }
    
    prediction = diabetes_models.predict_patient(new_patient)
    print(f"Вероятность диабета: {prediction['probability']:.2%}")
    print(f"Уровень риска: {prediction['risk_level']}")
    print(f"Рекомендация: {prediction['recommendation']}")
    
    return diabetes_models


if __name__ == "__main__":
    # Запуск тестирования
    test_models()