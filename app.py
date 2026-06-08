# ====================================
# Главный файл графического приложения
# ====================================

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Импорт модулей
from models import DiabetesModels
from visualizations import DiabetesVisualizer
from database import DatabaseManager
from encryption import DataEncryptor, PatientDataProtector, AuditLogger
from config import *

class DiabetesApp:
    # Главный класс приложения
    
    def __init__(self, root):
        self.root = root
        self.root.title("🏥 Система прогнозирования риска диабета v2.0")
        self.root.geometry("1400x1000")
        self.root.configure(bg='#f0f0f0')
        
        # Инициализация модулей
        self.db_manager = None
        self.encryptor = None
        self.protector = None
        self.audit_logger = None
        self.ml_models = None
        self.visualizer = None
        
        # Данные и результаты
        self.df = None
        self.results_df = None
        self.visualizer = DiabetesVisualizer()
        
        # Пытаемся инициализировать БД
        try:
            self.db_manager = DatabaseManager(db_type='sqlite', db_path=DATA_DIR / 'diabetes.db')
            print("База данных инициализирована")
        except Exception as e:
            print(f"Ошибка инициализации БД: {e}")
        
        # Пытаемся инициализировать шифрование
        try:
            self.encryptor = DataEncryptor(password="diabetes_secure_2024")
            self.protector = PatientDataProtector(self.encryptor)
            self.audit_logger = AuditLogger(LOGS_DIR / 'audit.log')
            print("Модуль шифрования инициализирован")
        except Exception as e:
            print(f"Ошибка инициализации шифрования: {e}")
        
        # Цветовая схема
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'light': '#ecf0f1',
            'dark': '#2c3e50'
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Настройка пользовательского интерфейса
        
        # Верхняя панель с заголовком
        self.create_header()
        
        # Панель инструментов
        self.create_toolbar()
        
        # Статусная строка
        self.create_statusbar()
        
        # Основная область с вкладками
        self.create_notebook()
        
        # Создание вкладок
        self.create_tabs()
        
    def create_header(self):
        # Создание Верхней панели с заголовком
        
        title_frame = tk.Frame(self.root, bg=self.colors['primary'], height=80)
        title_frame.pack(fill='x')
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="ПРОГНОЗИРОВАНИЕ РИСКА ДИАБЕТА", 
            font=('Arial', 24, 'bold'), 
            fg='white', 
            bg=self.colors['primary']
        )
        title_label.pack(pady=20)
        
        subtitle_label = tk.Label(
            title_frame,
            text="Система поддержки принятия врачебных решений на основе машинного обучения",
            font=('Arial', 10),
            fg='#bdc3c7',
            bg=self.colors['primary']
        )
        subtitle_label.pack()
        
    def create_toolbar(self):
        # Создание панели инструментов
        toolbar_frame = tk.Frame(self.root, bg=self.colors['light'], height=60)
        toolbar_frame.pack(fill='x', pady=5)
        toolbar_frame.pack_propagate(False)
        
        btn_style = {'font': ('Arial', 11, 'bold'), 'width': 24, 'height': 1}
        
        # Кнопка загрузки из URL
        self.url_btn = tk.Button(
            toolbar_frame, 
            text="Загрузить из URL", 
            command=self.load_from_url,
            bg=self.colors['secondary'], 
            fg='white', 
            **btn_style
        )
        self.url_btn.pack(side='left', padx=10, pady=12)
        
        # Кнопка загрузки из файла
        self.file_btn = tk.Button(
            toolbar_frame, 
            text="Загрузить из Excel/CSV", 
            command=self.load_from_file,
            bg=self.colors['secondary'], 
            fg='white', 
            **btn_style
        )
        self.file_btn.pack(side='left', padx=5, pady=12)
        
        # Кнопка загрузки из БД
        self.db_btn = tk.Button(
            toolbar_frame, 
            text="Загрузить из БД", 
            command=self.load_from_db,
            bg=self.colors['secondary'], 
            fg='white', 
            **btn_style
        )
        self.db_btn.pack(side='left', padx=5, pady=12)
        
        # Кнопка обучения моделей
        self.train_btn = tk.Button(
            toolbar_frame, 
            text="Обучить модели", 
            command=self.train_models,
            bg=self.colors['success'], 
            fg='white', 
            **btn_style
        )
        self.train_btn.pack(side='left', padx=5, pady=12)
        self.train_btn.config(state='disabled')
        
        # Кнопка экспорта результатов
        self.export_btn = tk.Button(
            toolbar_frame, 
            text="Экспорт результатов", 
            command=self.export_results,
            bg=self.colors['warning'], 
            fg='white', 
            **btn_style
        )
        self.export_btn.pack(side='left', padx=5, pady=12)
        self.export_btn.config(state='disabled')
        
        # Кнопка сохранения в БД
        self.save_db_btn = tk.Button(
            toolbar_frame, 
            text="Сохранить в БД", 
            command=self.save_to_database,
            bg=self.colors['primary'], 
            fg='white', 
            **btn_style
        )
        self.save_db_btn.pack(side='left', padx=5, pady=12)
        self.save_db_btn.config(state='disabled')
        
        # Прогресс-бар
        self.progress = ttk.Progressbar(toolbar_frame, mode='indeterminate', length=200)
        self.progress.pack(side='right', padx=10, pady=12)
        self.progress.pack_forget()
        
    def create_statusbar(self):
        # Создание статусной строки
        self.status_var = tk.StringVar(value="Готов к работе. Загрузите данные для анализа.")
        status_bar = tk.Label(
            self.root, 
            textvariable=self.status_var, 
            bg=self.colors['dark'], 
            fg='white', 
            font=('Arial', 10), 
            anchor='w'
        )
        status_bar.pack(side='bottom', fill='x')
        
    def create_notebook(self):
        # Создание области с вкладками
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
    def create_tabs(self):
        """Создание всех вкладок"""
        self.tabs = {}
        tab_names = {
            'data': 'Данные',
            'roc': 'ROC-кривые',
            'features': 'Важность признаков',
            'calibration': 'Калибровка',
            'threshold': 'Оптимальный порог',
            'results': 'Результаты',
            'metrics': 'Сравнение моделей',
            'prediction': 'Прогноз пациента',
        }
        
        for tab_key, tab_name in tab_names.items():
            frame = ttk.Frame(self.notebook)
            self.notebook.add(frame, text=tab_name)
            self.tabs[tab_key] = frame
            
        self.init_data_tab()
        self.init_results_tab()
        self.init_prediction_tab()

    def init_data_tab(self):
        # Инициализация вкладки с данными
        frame = self.tabs['data']
        
        control_frame = tk.Frame(frame)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(control_frame, text="Просмотр данных:", font=('Arial', 12, 'bold')).pack(side='left')
        
        tree_frame = tk.Frame(frame)
        tree_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        scrollbar_y = tk.Scrollbar(tree_frame)
        scrollbar_y.pack(side='right', fill='y')
        
        scrollbar_x = tk.Scrollbar(tree_frame, orient='horizontal')
        scrollbar_x.pack(side='bottom', fill='x')
        
        self.data_tree = ttk.Treeview(
            tree_frame, 
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set
        )
        self.data_tree.pack(fill='both', expand=True)
        
        scrollbar_y.config(command=self.data_tree.yview)
        scrollbar_x.config(command=self.data_tree.xview)
        
        self.data_info_label = tk.Label(frame, text="", font=('Arial', 10), fg='blue', anchor='w')
        self.data_info_label.pack(fill='x', padx=5, pady=5)
        
    def init_results_tab(self):
        # Инициализация вкладки с результатами
        frame = self.tabs['results']
        
        text_frame = tk.Frame(frame)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.results_text = tk.Text(
            text_frame, 
            wrap='word', 
            font=('Courier', 10),
            yscrollcommand=scrollbar.set
        )
        self.results_text.pack(fill='both', expand=True)
        scrollbar.config(command=self.results_text.yview)
        
        btn_frame = tk.Frame(frame)
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="Копировать результаты", command=self.copy_results,
                 bg='#95a5a6', fg='white', font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        
        tk.Button(btn_frame, text="Сохранить отчет", command=self.save_report,
                 bg='#3498db', fg='white', font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        
    def start_loading(self):
        # Начать индикацию загрузки
        self.progress.pack(side='right', padx=10, pady=12)
        self.progress.start(10)
        self.url_btn.config(state='disabled')
        self.file_btn.config(state='disabled')
        self.db_btn.config(state='disabled')
        
    def stop_loading(self):
        # Остановить индикацию загрузки
        self.progress.stop()
        self.progress.pack_forget()
        self.url_btn.config(state='normal')
        self.file_btn.config(state='normal')
        self.db_btn.config(state='normal')
        
    def load_from_url(self):
        # Загрузка данных из URL
        dialog = tk.Toplevel(self.root)
        dialog.title("Загрузка из URL")
        dialog.geometry("550x180")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Введите URL данных:", font=('Arial', 11), bg='#f0f0f0').pack(pady=10)
        
        url_entry = tk.Entry(dialog, width=70, font=('Arial', 10))
        url_entry.pack(pady=5)
        url_entry.insert(0, DATA_CONFIG['default_url'])
        
        def load():
            url = url_entry.get()
            if url:
                self.start_loading()
                try:
                    columns = DATA_CONFIG['data_columns']
                    self.df = pd.read_csv(url, names=columns)
                    self.update_data_display()
                    self.status_var.set(f"Данные загружены из URL. Размер: {self.df.shape}")
                    dialog.destroy()
                    
                    # Активируем кнопки
                    self.train_btn.config(state='normal')
                    
                    # Логируем
                    if self.audit_logger:
                        self.audit_logger.log_access(
                            os.getenv('USER', 'unknown'), 'LOAD', 'URL', 'SUCCESS',
                            {'url': url[:50], 'shape': self.df.shape}
                        )
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось загрузить данные:\n{str(e)}")
                finally:
                    self.stop_loading()
        
        tk.Button(dialog, text="Загрузить", command=load, 
                 bg='#3498db', fg='white', font=('Arial', 10, 'bold'), width=15).pack(pady=10)
        
    def load_from_file(self):
        # Загрузка данных из файла
        file_path = filedialog.askopenfilename(
            title="Выберите файл данных",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        
        if file_path:
            self.start_loading()
            try:
                if file_path.endswith('.csv'):
                    self.df = pd.read_csv(file_path)
                else:
                    self.df = pd.read_excel(file_path)
                
                self.update_data_display()
                self.status_var.set(f"Данные загружены из файла. Размер: {self.df.shape}")
                self.train_btn.config(state='normal')
                
                if self.audit_logger:
                    self.audit_logger.log_access(
                        os.getenv('USER', 'unknown'), 'LOAD', 'FILE', 'SUCCESS',
                        {'file': file_path, 'shape': self.df.shape}
                    )
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{str(e)}")
            finally:
                self.stop_loading()
                
    def load_from_db(self):
        # Загрузка данных из базы данных
        if not self.db_manager:
            messagebox.showwarning("Предупреждение", "База данных не инициализирована")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Загрузка из БД")
        dialog.geometry("400x200")
        dialog.configure(bg='#f0f0f0')
        
        tk.Label(dialog, text="Введите хеш пациента:", font=('Arial', 11), bg='#f0f0f0').pack(pady=10)
        
        hash_entry = tk.Entry(dialog, width=40, font=('Arial', 10))
        hash_entry.pack(pady=5)
        
        def load():
            patient_hash = hash_entry.get()
            if patient_hash:
                try:
                    self.df = self.db_manager.load_patient_data(patient_hash)
                    self.update_data_display()
                    self.status_var.set(f"Данные загружены из БД. Размер: {self.df.shape}")
                    self.train_btn.config(state='normal')
                    dialog.destroy()
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось загрузить из БД:\n{str(e)}")
        
        tk.Button(dialog, text="Загрузить", command=load, 
                 bg='#3498db', fg='white', font=('Arial', 10, 'bold')).pack(pady=10)
        
    def update_data_display(self):
        # Обновление отображения данных
        if self.df is not None:
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            columns = list(self.df.columns)
            self.data_tree['columns'] = columns
            self.data_tree['show'] = 'headings'
            
            for col in columns:
                self.data_tree.heading(col, text=col)
                self.data_tree.column(col, width=100)
            
            for idx, row in self.df.head(100).iterrows():
                values = [str(v)[:50] for v in row.values]
                self.data_tree.insert('', 'end', values=values)
            
            info_text = f"Размер: {self.df.shape[0]} строк × {self.df.shape[1]} колонок | "
            if 'Outcome' in self.df.columns:
                counts = self.df['Outcome'].value_counts().to_dict()
                info_text += f"Баланс классов: {counts}"
            
            self.data_info_label.config(text=info_text)
            self.notebook.select(self.tabs['data'])
            
    def train_models(self):
        # Обучение моделей машинного обучения
        if self.df is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
        
        if 'Outcome' not in self.df.columns:
            messagebox.showerror("Ошибка", "В данных отсутствует целевая переменная 'Outcome'")
            return
        
        self.start_loading()
        self.status_var.set("Обучение моделей... Пожалуйста, подождите")
        self.root.update()
        
        try:
            # Инициализируем модели
            self.ml_models = DiabetesModels()
            self.ml_models.load_data(self.df)
            self.ml_models.preprocess_data()
            self.ml_models.init_models()
            
            # Обучаем
            self.results_df = self.ml_models.train_all_models()
            
            # Сохраняем результаты в БД
            if self.db_manager:
                for _, row in self.results_df.iterrows():
                    metrics = {
                        'accuracy': row['Accuracy'],
                        'precision': row['Precision'],
                        'recall': row['Recall'],
                        'f1_score': row['F1-Score'],
                        'roc_auc': row['ROC-AUC'],
                        'cv_score': row.get('CV-AUC', row['ROC-AUC']),
                        'training_samples': len(self.ml_models.X_train_scaled),
                        'test_samples': len(self.ml_models.X_test_scaled)
                    }
                    self.db_manager.save_model_results(row['Model'], metrics)
            
            # Отображаем результаты
            self.display_results()
            
            # Строим графики
            self.plot_all_charts()
            
            self.status_var.set(f"Модели обучены! Лучшая: {self.ml_models.best_model_name}")
            self.export_btn.config(state='normal')
            self.save_db_btn.config(state='normal')
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при обучении:\n{str(e)}")
            self.status_var.set(f"Ошибка: {str(e)[:100]}")
        finally:
            self.stop_loading()
      
    def plot_all_charts(self):
        # Построение всех графиков
        if not self.ml_models:
            return
        
        # ROC кривые
        self.visualizer.create_roc_curves(
            self.ml_models.y_test, 
            self.ml_models.probabilities,
            embed=True, 
            parent_frame=self.tabs['roc']
        )
        
        # Feature importance
        importance_df = self.ml_models.get_feature_importance()
        if importance_df is not None:
            self.visualizer.create_feature_importance(
                importance_df['Feature'].values,
                importance_df['Importance'].values,
                embed=True,
                parent_frame=self.tabs['features']
            )
        
        # Calibration curves
        self.visualizer.create_calibration_curves(
            self.ml_models.y_test,
            self.ml_models.probabilities,
            embed=True,
            parent_frame=self.tabs['calibration']
        )
        
        # Threshold analysis
        fig, info = self.visualizer.create_threshold_analysis(
            self.ml_models.y_test,
            self.ml_models.probabilities,
            self.ml_models.best_model_name,
            embed=True,
            parent_frame=self.tabs['threshold']
        )
        
        # Model comparison
        self.visualizer.create_model_comparison(
            self.results_df,
            embed=True,
            parent_frame=self.tabs['metrics']
        )
        
    def display_results(self):
        # Отображение текстовых результатов
        if self.results_df is None:
            return
        
        self.results_text.delete('1.0', tk.END)
        
        output = "="*70 + "\n"
        output += "РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ РИСКА ДИАБЕТА\n"
        output += "="*70 + "\n\n"
        
        output += f"РАЗМЕР ВЫБОРКИ:\n"
        output += f"   • Всего пациентов: {len(self.df)}\n"
        output += f"   • Обучающая выборка: {len(self.ml_models.X_train_scaled)}\n"
        output += f"   • Тестовая выборка: {len(self.ml_models.X_test_scaled)}\n\n"
        
        output += "СРАВНЕНИЕ МОДЕЛЕЙ:\n"
        output += "-"*70 + "\n"
        
        for _, row in self.results_df.iterrows():
            output += f"\n {row['Model']}:\n"
            output += f"   • Accuracy:  {row['Accuracy']:.4f}\n"
            output += f"   • Precision: {row['Precision']:.4f}\n"
            output += f"   • Recall:    {row['Recall']:.4f}\n"
            output += f"   • F1-Score:  {row['F1-Score']:.4f}\n"
            output += f"   • ROC-AUC:   {row['ROC-AUC']:.4f}\n"
        
        output += "\n" + "="*70 + "\n"
        output += f"ЛУЧШАЯ МОДЕЛЬ: {self.ml_models.best_model_name}\n"
        output += "="*70 + "\n"
        
        # Оптимальный порог
        optimal = self.ml_models.get_optimal_threshold()
        if optimal:
            output += f"\nОптимальный порог: {optimal['optimal_threshold']:.3f}\n"
            output += f"   Precision: {optimal['precision']:.4f}\n"
            output += f"   Recall:    {optimal['recall']:.4f}\n"
            output += f"   F1-score:  {optimal['f1_score']:.4f}\n"
        
        self.results_text.insert('1.0', output)

    def save_to_database(self):
        
        # Сохранение данных в базу данных (без предсказаний)
        if self.df is None:
            messagebox.showwarning("Предупреждение", "Нет данных для сохранения!")
            return
        
        if not self.db_manager:
            messagebox.showwarning("Предупреждение", "База данных не инициализирована")
            return
        
        try:
            self.start_loading()
            
            # Анонимизируем данные
            if self.protector:
                df_to_save = self.protector.anonymize_patient_id(self.df.copy())
            else:
                df_to_save = self.df.copy()
            
            # Сохраняем только данные пациентов (без предсказаний)
            patient_hash = self.db_manager.save_patient_data(df_to_save)
            
            messagebox.showinfo("Успех", f"Данные сохранены в БД\nХеш пациента: {patient_hash[:16]}...")
            self.status_var.set(f"Данные сохранены в БД. Хеш: {patient_hash[:16]}...")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить в БД:\n{str(e)}")
        finally:
            self.stop_loading()
                
        def export_results(self):
            # Экспорт результатов в файл
            if self.results_df is None:
                messagebox.showwarning("Предупреждение", "Нет результатов для экспорта!")
                return
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("HTML files", "*.html")]
            )
            
            if file_path:
                try:
                    if file_path.endswith('.csv'):
                        self.results_df.to_csv(file_path, index=False)
                    elif file_path.endswith('.html'):
                        self.results_df.to_html(file_path, index=False)
                    else:
                        self.results_df.to_excel(file_path, index=False)
                    
                    self.status_var.set(f"Результаты экспортированы в {file_path}")
                    messagebox.showinfo("Успех", f"Результаты сохранены в:\n{file_path}")
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось экспортировать: {str(e)}")
                
    def copy_results(self):
        # Копирование результатов в буфер обмена
        text = self.results_text.get('1.0', tk.END)
        if text.strip():
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_var.set("📋 Результаты скопированы в буфер обмена")
            
    def save_report(self):
        # Сохранение отчета в файл
        text = self.results_text.get('1.0', tk.END)
        if text.strip():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if file_path:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    self.status_var.set(f"Отчет сохранен в {file_path}")
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Не удалось сохранить отчет: {str(e)}")

    def export_results(self):
        # Экспорт результатов в файл
        if self.results_df is None:
            messagebox.showwarning("Предупреждение", "Нет результатов для экспорта!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[
                ("Excel files", "*.xlsx"), 
                ("CSV files", "*.csv"),
                ("HTML files", "*.html")
            ]
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.results_df.to_csv(file_path, index=False)
                elif file_path.endswith('.html'):
                    self.results_df.to_html(file_path, index=False)
                else:
                    self.results_df.to_excel(file_path, index=False)
                
                self.status_var.set(f"Результаты экспортированы в {file_path}")
                messagebox.showinfo("Успех", f"Результаты сохранены в:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось экспортировать: {str(e)}")                

    def init_prediction_tab(self):
        """Вкладка для ручного ввода данных пациента и получения прогноза"""
        frame = self.tabs['prediction']
        
        # Заголовок
        tk.Label(
            frame, 
            text="ПРОГНОЗ РИСКА ДИАБЕТА ДЛЯ ПАЦИЕНТА", 
            font=('Arial', 16, 'bold'),
            fg='#2c3e50'
        ).pack(pady=15)
        
        tk.Label(
            frame,
            text="Введите клинические показатели пациента для оценки риска развития диабета 2 типа",
            font=('Arial', 10),
            fg='#7f8c8d'
        ).pack(pady=(0, 20))
        
        # Основная рамка (две колонки)
        main_frame = tk.Frame(frame)
        main_frame.pack(fill='both', expand=True, padx=30, pady=10)
        
        # ЛЕВАЯ КОЛОНКА - поля ввода
        left_frame = tk.LabelFrame(main_frame, text="Клинические параметры пациента", font=('Arial', 12, 'bold'))
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        
        # ПРАВАЯ КОЛОНКА - результат прогноза
        right_frame = tk.LabelFrame(main_frame, text="Результат прогнозирования", font=('Arial', 12, 'bold'))
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        # ============================================
        # ЛЕВАЯ КОЛОНКА: ПОЛЯ ВВОДА С ПОЯСНЕНИЯМИ
        # ============================================
        
        # Словарь для хранения переменных ввода
        self.prediction_vars = {}
        
        # Данные о признаках: ключ, название, единица измерения, норма, подробное пояснение
        features_info = [
            ('Pregnancies', 'Количество беременностей', 'кол-во', '0-5', 
            'Пояснение: количество предыдущих беременностей у пациентки. Беременность увеличивает нагрузку на поджелудочную железу. Более 5 беременностей считается фактором риска.'),
            
            ('Glucose', 'Уровень глюкозы', 'мг/дл', '70-99', 
            'Пояснение: концентрация глюкозы в плазме крови через 2 часа после еды. Норма: 70-99 мг/дл. 100-125 - преддиабет, 126 и выше - диабет.'),
            
            ('BloodPressure', 'Диастолическое давление', 'мм рт.ст.', '60-80', 
            'Пояснение: нижнее давление (в момент расслабления сердца). Норма: 60-80 мм рт.ст. Повышение давления (90+) увеличивает риск диабета.'),
            
            ('SkinThickness', 'Толщина кожной складки', 'мм', '10-30', 
            'Пояснение: толщина кожной складки в области трицепса. Показатель подкожного жира. Норма: 10-30 мм. Высокие значения указывают на ожирение.'),
            
            ('Insulin', 'Уровень инсулина', 'мкЕд/мл', '16-180', 
            'Пояснение: уровень инсулина в сыворотке крови через 2 часа после нагрузки. Норма: 16-180 мкЕд/мл. Отклонения могут указывать на инсулинорезистентность.'),
            
            ('BMI', 'Индекс массы тела', 'кг/м²', '18.5-25', 
            'Пояснение: ИМТ = вес(кг) / рост²(м²). 18.5-25 - норма, 25-30 - избыточный вес, 30+ - ожирение. Ожирение - главный фактор риска диабета.'),
            
            ('DiabetesPedigreeFunction', 'Наследственность (DPF)', '', '0-0.5', 
            'Пояснение: коэффициент наследственной предрасположенности. 0 - нет родственников с диабетом, 0.5 - один близкий родственник, 1.0+ - сильная наследственность.'),
            
            ('Age', 'Возраст', 'лет', '20-60', 
            'Пояснение: возраст пациента. Риск развития диабета 2 типа значительно возрастает после 45 лет. Чем старше пациент, тем выше риск.')
        ]
        
        # Создание полей ввода
        for key, label, unit, norm, explanation in features_info:
            # Рамка для одного параметра
            param_frame = tk.Frame(left_frame, relief='groove', bd=1)
            param_frame.pack(fill='x', pady=5, padx=6)
            
            # Верхняя строка: название, поле ввода, норма
            top_frame = tk.Frame(param_frame)
            top_frame.pack(fill='x', padx=5, pady=5)
            
            # Название поля с единицей измерения
            label_text = f"{label} ({unit})" if unit else label
            tk.Label(top_frame, text=label_text, width=35, anchor='w', font=('Arial', 10, 'bold')).pack(side='left')
            
            # Поле ввода
            var = tk.StringVar()
            entry = tk.Entry(top_frame, textvariable=var, width=12, font=('Arial', 10))
            entry.pack(side='left', padx=15)
            
            # Метка с нормой
            tk.Label(top_frame, text=f"норма: {norm}", fg='green', font=('Arial', 9)).pack(side='left', padx=5)
            
            # Сохраняем переменную
            self.prediction_vars[key] = var
            
            # Нижняя строка: пояснение (серым цветом)
            bottom_frame = tk.Frame(param_frame)
            bottom_frame.pack(fill='x', padx=5, pady=(0, 5))
            
            tk.Label(bottom_frame, text=explanation, fg='gray', font=('Arial', 8), anchor='w', wraplength=350, justify='left').pack(fill='x')
        
        # Кнопка очистки всех полей
        def clear_fields():
            for var in self.prediction_vars.values():
                var.set("")
            # Очистить результаты прогноза
            self.prob_label.config(text="---")
            self.pred_label.config(text="---")
            self.risk_label.config(text="---")
            self.recommend_label.config(text="Заполните данные пациента и нажмите 'Выполнить прогноз'")
            self.recommend_label.config(fg='#7f8c8d')
        
        tk.Button(
            left_frame, 
            text="Очистить все поля", 
            command=clear_fields,
            bg='#95a5a6', 
            fg='white',
            font=('Arial', 10),
            width=20
        ).pack(pady=15)
        
        # ============================================
        # ПРАВАЯ КОЛОНКА: РЕЗУЛЬТАТЫ ПРОГНОЗА
        # ============================================
        
        # Рамка для отображения результата
        result_frame = tk.Frame(right_frame, bg='#ecf0f1', relief='groove', bd=2)
        result_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Метка для вероятности
        tk.Label(result_frame, text="Вероятность диабета:", font=('Arial', 12), bg='#ecf0f1').pack(pady=(20, 5))
        self.prob_label = tk.Label(result_frame, text="---", font=('Arial', 28, 'bold'), fg='#2c3e50', bg='#ecf0f1')
        self.prob_label.pack()
        
        # Метка для предсказания
        tk.Label(result_frame, text="Предсказание:", font=('Arial', 12), bg='#ecf0f1').pack(pady=(15, 5))
        self.pred_label = tk.Label(result_frame, text="---", font=('Arial', 18, 'bold'), bg='#ecf0f1')
        self.pred_label.pack()
        
        # Метка для уровня риска
        tk.Label(result_frame, text="Уровень риска:", font=('Arial', 12), bg='#ecf0f1').pack(pady=(15, 5))
        self.risk_label = tk.Label(result_frame, text="---", font=('Arial', 14, 'bold'), bg='#ecf0f1')
        self.risk_label.pack()
        
        # Рекомендации
        tk.Label(result_frame, text="Рекомендации:", font=('Arial', 12, 'bold'), bg='#ecf0f1').pack(pady=(20, 5))
        self.recommend_label = tk.Label(
            result_frame, 
            text="Заполните данные пациента и нажмите 'Выполнить прогноз'", 
            font=('Arial', 10),
            wraplength=350,
            fg='#7f8c8d',
            bg='#ecf0f1'
        )
        self.recommend_label.pack(pady=10)
        
        # Кнопка прогноза
        tk.Button(
            right_frame,
            text="ВЫПОЛНИТЬ ПРОГНОЗ",
            command=self.predict_single_patient,
            bg='#27ae60',
            fg='white',
            font=('Arial', 12, 'bold'),
            height=2,
            width=25
        ).pack(pady=20, fill='x', padx=15)
        
        # Информация об используемой модели и пороге
        self.model_info_label = tk.Label(
            right_frame,
            text="Модель будет выбрана после обучения | Порог: 0.35 (медицинский)",
            font=('Arial', 9),
            fg='#7f8c8d'
        )
        self.model_info_label.pack(pady=5)

    def predict_single_patient(self):
        # Обработчик нажатия кнопки прогноза
        
        # ПРОВЕРКА: обучены ли модели?
        if self.ml_models is None or self.ml_models.best_model is None:
            messagebox.showwarning("Предупреждение", 
                "Сначала обучите модели на вкладке 'Данные'!")
            return
        
        try:
            # СБОР ДАННЫХ ИЗ ПОЛЕЙ ВВОДА
            feature_order = ['Pregnancies', 'Glucose', 'BloodPressure', 
                            'SkinThickness', 'Insulin', 'BMI', 
                            'DiabetesPedigreeFunction', 'Age']
            
            patient_data = []
            for feature in feature_order:
                value_str = self.prediction_vars[feature].get()
                
                if not value_str.strip():
                    messagebox.showerror("Ошибка", f"Заполните поле: {feature}")
                    return
                
                try:
                    value = float(value_str)
                    if value < 0:
                        messagebox.showerror("Ошибка", f"Значение {feature} не может быть отрицательным")
                        return
                    patient_data.append(value)
                except ValueError:
                    messagebox.showerror("Ошибка", f"В поле {feature} должно быть число")
                    return
            
            # ВЫЗОВ МЕТОДА ПРЕДСКАЗАНИЯ ИЗ models.py
            result = self.ml_models.predict_patient(patient_data)
            
            # ОТОБРАЖЕНИЕ РЕЗУЛЬТАТА
            
            # Вероятность
            self.prob_label.config(text=f"{result['probability']*100:.1f}%")
            
            # Предсказание
            if result['prediction'] == 1:
                self.pred_label.config(text="⚠️ ДИАБЕТ", fg="#e74c3c")
            else:
                self.pred_label.config(text="✅ НЕТ ДИАБЕТА", fg="#27ae60")
            
            # Уровень риска
            risk_colors = {"Низкий": "#27ae60", "Средний": "#f39c12", "Высокий": "#e74c3c"}
            self.risk_label.config(text=result['risk_level'], fg=risk_colors[result['risk_level']])
            
            # Рекомендации
            self.recommend_label.config(text=result['recommendation'])
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось выполнить прогноз:\n{str(e)}")

    def run(self):
        # Запуск приложения
        self.root.mainloop()


def main():
    # Точка входа в приложение
    root = tk.Tk()
    app = DiabetesApp(root)
    app.run()


if __name__ == "__main__":
    main()