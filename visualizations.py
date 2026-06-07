# visualizations.py
# Модуль для создания графиков и визуализаций

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Настройка стилей
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9


class DiabetesVisualizer:
    # Класс для создания визуализаций прогнозирования диабета
    
    def __init__(self, style='darkgrid'):
        
        # Инициализация визуализатора
        self.style = style
        sns.set_style(style)
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'purple': '#9b59b6',
            'orange': '#e67e22',
            'cyan': '#1abc9c'
        }
        
    def create_roc_curves(self, y_test, probabilities, model_names=None, 
                         figsize=(10, 8), embed=False, parent_frame=None):
        
        # Создание ROC-кривых для всех моделей
        fig = Figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        
        # Цветовая схема для моделей
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
        
        for idx, (name, probs) in enumerate(probabilities.items()):
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc = np.trapezoid(tpr, fpr)  # Площадь под кривой
            
            color = colors[idx % len(colors)]
            ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', 
                   linewidth=2, color=color)
        
        # Диагональная линия (случайный классификатор)
        ax.plot([0, 1], [0, 1], 'k--', label='Случайный классификатор', 
               linewidth=1, alpha=0.7)
        
        # Настройки графика
        ax.set_xlabel('False Positive Rate (1 - Специфичность)', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Positive Rate (Чувствительность)', fontsize=11, fontweight='bold')
        ax.set_title('ROC-кривые моделей прогнозирования диабета', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])
        
        # Добавляем аннотацию с лучшей моделью
        best_auc = 0
        best_name = ""
        for name, probs in probabilities.items():
            fpr, tpr, _ = roc_curve(y_test, probs)
            auc = np.trapezoid(tpr, fpr)
            if auc > best_auc:
                best_auc = auc
                best_name = name
        
        ax.text(0.05, 0.05, f'Лучшая модель: {best_name}\n   AUC = {best_auc:.3f}', 
               transform=ax.transAxes, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        fig.tight_layout()
        
        if embed and parent_frame:
            return self.embed_figure(fig, parent_frame)
        return fig
    
    def create_feature_importance(self, feature_names, importances, 
                                 top_n=10, figsize=(10, 6), 
                                 embed=False, parent_frame=None):
        
        # Создание графика важности признаков
        fig = Figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        
        # Сортируем по важности
        indices = np.argsort(importances)[-top_n:]
        sorted_importances = importances[indices]
        sorted_features = [feature_names[i] for i in indices]
        
        # Создаем горизонтальную гистограмму
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_features)))
        bars = ax.barh(range(len(sorted_features)), sorted_importances, color=colors)
        
        # Добавляем значения на бары
        for i, (bar, value) in enumerate(zip(bars, sorted_importances)):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', va='center', fontsize=9)
        
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Важность признака', fontsize=11, fontweight='bold')
        ax.set_title('Важность признаков для прогнозирования диабета\n(Random Forest)', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim([0, max(sorted_importances) * 1.1])
        
        # Инвертируем ось Y для лучшего отображения
        ax.invert_yaxis()
        
        fig.tight_layout()
        
        if embed and parent_frame:
            return self.embed_figure(fig, parent_frame)
        return fig
    
    def create_calibration_curves(self, y_test, probabilities, 
                                 n_bins=10, figsize=(10, 6),
                                 embed=False, parent_frame=None):
        
        # Создание калибровочных кривых
        fig = Figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        for idx, (name, probs) in enumerate(probabilities.items()):
            prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=n_bins)
            color = colors[idx % len(colors)]
            ax.plot(prob_pred, prob_true, marker='o', label=name, 
                   linewidth=2, markersize=6, color=color)
        
        # Идеально калиброванная линия
        ax.plot([0, 1], [0, 1], 'k--', label='Идеально калиброванная', 
               linewidth=1, alpha=0.7)
        
        ax.set_xlabel('Средняя предсказанная вероятность', fontsize=11, fontweight='bold')
        ax.set_ylabel('Доля положительных исходов', fontsize=11, fontweight='bold')
        ax.set_title('Калибровочные кривые моделей', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        
        fig.tight_layout()
        
        if embed and parent_frame:
            return self.embed_figure(fig, parent_frame)
        return fig
    
    def create_threshold_analysis(self, y_test, probabilities, model_name=None,
                                 figsize=(10, 6), embed=False, parent_frame=None):
        
        # Анализ оптимального порога принятия решения
        if model_name is None:
            model_name = list(probabilities.keys())[0]
        
        y_scores = probabilities[model_name]
        
        thresholds = np.arange(0.01, 0.99, 0.01)
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
            recall_scores.append(recall_score(y_test, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        
        # Находим оптимальный порог
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        optimal_precision = precision_scores[optimal_idx]
        optimal_recall = recall_scores[optimal_idx]
        
        fig = Figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        
        ax.plot(thresholds, precision_scores, label='Точность (Precision)', 
               linewidth=2, color='#3498db')
        ax.plot(thresholds, recall_scores, label='Полнота (Recall)', 
               linewidth=2, color='#2ecc71')
        ax.plot(thresholds, f1_scores, label='F1-мера', 
               linewidth=2, color='#e74c3c')
        
        # Вертикальная линия оптимального порога
        ax.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax.text(optimal_threshold + 0.02, 0.5, f'Оптимальный порог\n{optimal_threshold:.2f}', 
               fontsize=9, color='red', fontweight='bold')
        
        # Отмечаем точку максимума F1
        ax.plot(optimal_threshold, optimal_f1, 'ro', markersize=10, label=f'Max F1 = {optimal_f1:.3f}')
        
        ax.set_xlabel('Порог принятия решения', fontsize=11, fontweight='bold')
        ax.set_ylabel('Значение метрики', fontsize=11, fontweight='bold')
        ax.set_title(f'Выбор оптимального порога\nМодель: {model_name}', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        
        fig.tight_layout()
        
        optimal_info = {
            'model_name': model_name,
            'optimal_threshold': optimal_threshold,
            'precision': optimal_precision,
            'recall': optimal_recall,
            'f1_score': optimal_f1,
            'thresholds': thresholds,
            'precision_scores': precision_scores,
            'recall_scores': recall_scores,
            'f1_scores': f1_scores
        }
        
        if embed and parent_frame:
            return self.embed_figure(fig, parent_frame), optimal_info
        return fig, optimal_info
    
    def create_confusion_matrix(self, y_test, y_pred, model_name,
                               figsize=(8, 6), embed=False, parent_frame=None):
        
        # Создание матрицы ошибок
        cm = confusion_matrix(y_test, y_pred)
        
        fig = Figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        
        # Создаем тепловую карту
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Настройка меток
        classes = ['Нет диабета (0)', 'Диабет (1)']
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        
        # Добавляем значения в ячейки
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                       color='white' if cm[i, j] > cm.max() / 2 else 'black',
                       fontsize=16, fontweight='bold')
        
        ax.set_xlabel('Предсказанный класс', fontsize=11, fontweight='bold')
        ax.set_ylabel('Истинный класс', fontsize=11, fontweight='bold')
        ax.set_title(f'Матрица ошибок\n{model_name}', fontsize=13, fontweight='bold', pad=15)
        
        # Добавляем метрики
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nSpecificity: {specificity:.3f}'
        ax.text(1.1, 0.5, metrics_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='center', bbox=dict(boxstyle="round", facecolor="lightyellow"))
        
        fig.tight_layout()
        
        if embed and parent_frame:
            return self.embed_figure(fig, parent_frame)
        return fig
    
    def create_pr_curve(self, y_test, probabilities, model_name=None,
                       figsize=(8, 6), embed=False, parent_frame=None):
        
        # Создание Precision-Recall кривой
        if model_name is None:
            model_name = list(probabilities.keys())[0]
        
        y_scores = probabilities[model_name]
        precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
        
        fig = Figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        
        ax.plot(recall, precision, linewidth=2, color='#3498db')
        ax.fill_between(recall, precision, alpha=0.3, color='#3498db')
        
        # Находим лучший баланс
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores[:-1])  # -1 потому что thresholds на 1 меньше
        best_precision = precision[best_idx]
        best_recall = recall[best_idx]
        
        ax.plot(best_recall, best_precision, 'ro', markersize=10, 
               label=f'Best F1 = {f1_scores[best_idx]:.3f}')
        
        ax.set_xlabel('Recall (Полнота)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Precision (Точность)', fontsize=11, fontweight='bold')
        ax.set_title(f'Precision-Recall кривая\n{model_name}', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        
        fig.tight_layout()
        
        if embed and parent_frame:
            return self.embed_figure(fig, parent_frame)
        return fig
    
    def create_distribution_plot(self, df, feature, target='Outcome',
                                figsize=(10, 6), embed=False, parent_frame=None):
        
        # Создание графика распределения признака по классам
        fig = Figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        
        # Разделяем по классам
        class0 = df[df[target] == 0][feature]
        class1 = df[df[target] == 1][feature]
        
        # Гистограммы
        ax.hist(class0, bins=20, alpha=0.5, label='Нет диабета', color='#3498db', edgecolor='black')
        ax.hist(class1, bins=20, alpha=0.5, label='Диабет', color='#e74c3c', edgecolor='black')
        
        # Добавляем средние значения
        ax.axvline(class0.mean(), color='#3498db', linestyle='--', linewidth=2, 
                  label=f'Среднее (нет): {class0.mean():.1f}')
        ax.axvline(class1.mean(), color='#e74c3c', linestyle='--', linewidth=2,
                  label=f'Среднее (диабет): {class1.mean():.1f}')
        
        ax.set_xlabel(feature, fontsize=11, fontweight='bold')
        ax.set_ylabel('Частота', fontsize=11, fontweight='bold')
        ax.set_title(f'Распределение признака "{feature}"\nпо классам диабета', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        
        if embed and parent_frame:
            return self.embed_figure(fig, parent_frame)
        return fig
    
    def create_correlation_matrix(self, df, figsize=(12, 10),
                                 embed=False, parent_frame=None):
        
        # Создание матрицы корреляций
        fig = Figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        
        # Вычисляем корреляцию
        corr = df.corr()
        
        # Создаем тепловую карту
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        
        # Настройка меток
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(corr.columns, fontsize=8)
        
        # Добавляем значения
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black" if abs(corr.iloc[i, j]) < 0.5 else "white",
                             fontsize=8)
        
        ax.set_title('Матрица корреляций признаков', fontsize=13, fontweight='bold', pad=15)
        
        # Добавляем colorbar
        fig.colorbar(im, ax=ax, label='Коэффициент корреляции')
        
        fig.tight_layout()
        
        if embed and parent_frame:
            return self.embed_figure(fig, parent_frame)
        return fig
    
    def create_model_comparison(self, results_df, figsize=(12, 6),
                               embed=False, parent_frame=None):
        
        # Создание графика сравнения моделей
        fig = Figure(figsize=figsize, dpi=100)
        ax = fig.add_subplot(111)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        models = results_df['Model'].values
        
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        for idx, model in enumerate(models):
            values = [results_df.loc[results_df['Model'] == model, metric].values[0] 
                     for metric in metrics]
            offset = (idx - len(models)/2) * width + width/2
            bars = ax.bar(x + offset, values, width, label=model, color=colors[idx % len(colors)])
            
            # Добавляем значения на бары
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Метрики', fontsize=11, fontweight='bold')
        ax.set_ylabel('Значение', fontsize=11, fontweight='bold')
        ax.set_title('Сравнение моделей прогнозирования диабета', 
                    fontsize=13, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        
        if embed and parent_frame:
            return self.embed_figure(fig, parent_frame)
        return fig
    
    def embed_figure(self, figure, parent_frame):
        
        # Встраивание графика в tkinter окно
        # Очищаем родительский фрейм
        for widget in parent_frame.winfo_children():
            widget.destroy()
        
        # Создаем canvas
        canvas = FigureCanvasTkAgg(figure, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Добавляем toolbar (опционально)
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, parent_frame)
        toolbar.update()
        
        return canvas


# Функция для тестирования модуля
def test_visualizer():
    # Тестовая функция для проверки визуализаций
    print("="*60)
    print("ТЕСТИРОВАНИЕ МОДУЛЯ ВИЗУАЛИЗАЦИЙ")
    print("="*60)
    
    # Создаем тестовые данные
    np.random.seed(42)
    y_test = np.random.randint(0, 2, 100)
    probabilities = {
        'Logistic Regression': np.random.rand(100),
        'Random Forest': np.random.rand(100),
        'Gradient Boosting': np.random.rand(100)
    }
    
    # Создаем визуализатор
    viz = DiabetesVisualizer()
    
    # Тестируем все графики
    print("\nСоздание графиков...")
    
    # 1. ROC кривые
    fig_roc = viz.create_roc_curves(y_test, probabilities)
    fig_roc.savefig('test_roc_curves.png', dpi=150, bbox_inches='tight')
    print("ROC-кривые сохранены")
    
    # 2. Feature importance
    feature_names = ['Glucose', 'BMI', 'Age', 'Pregnancies', 'Insulin', 
                    'BloodPressure', 'SkinThickness', 'DiabetesPedigreeFunction']
    importances = np.random.rand(len(feature_names))
    fig_imp = viz.create_feature_importance(feature_names, importances)
    fig_imp.savefig('test_feature_importance.png', dpi=150, bbox_inches='tight')
    print("Важность признаков сохранена")
    
    # 3. Calibration curves
    fig_cal = viz.create_calibration_curves(y_test, probabilities)
    fig_cal.savefig('test_calibration.png', dpi=150, bbox_inches='tight')
    print("Калибровочные кривые сохранены")
    
    # 4. Threshold analysis
    fig_thresh, info = viz.create_threshold_analysis(y_test, probabilities)
    fig_thresh.savefig('test_threshold.png', dpi=150, bbox_inches='tight')
    print(f"Анализ порога сохранен (оптимум: {info['optimal_threshold']:.3f})")
    
    # 5. PR curve
    fig_pr = viz.create_pr_curve(y_test, probabilities)
    fig_pr.savefig('test_pr_curve.png', dpi=150, bbox_inches='tight')
    print("Precision-Recall кривая сохранена")
    
    # 6. Model comparison
    results_df = pd.DataFrame({
        'Model': ['LR', 'RF', 'GB'],
        'Accuracy': [0.75, 0.80, 0.78],
        'Precision': [0.72, 0.78, 0.76],
        'Recall': [0.70, 0.82, 0.75],
        'F1-Score': [0.71, 0.80, 0.75],
        'ROC-AUC': [0.78, 0.85, 0.82]
    })
    fig_comp = viz.create_model_comparison(results_df)
    fig_comp.savefig('test_model_comparison.png', dpi=150, bbox_inches='tight')
    print(" Сравнение моделей сохранено")
    
    print("\nТестирование завершено! Все графики сохранены в текущей директории.")
    
    # Закрываем все фигуры
    plt.close('all')


if __name__ == "__main__":
    test_visualizer()