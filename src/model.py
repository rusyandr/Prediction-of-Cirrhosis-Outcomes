#!/usr/bin/env python3
"""
Модель для предсказания выживаемости пациентов с циррозом
Модуль для обучения и предсказания с использованием классификатора CatBoost
"""

import os
import sys
import logging
import argparse
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import catboost as cb
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.samplers import TPESampler

def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """Настройка логгера с обработчиками для файла и консоли"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Обработчик для консоли (уровень INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # Обработчик для файла (уровень DEBUG) - если указан файл лога
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = logging.getLogger(__name__)


class CirrhosisDataPreprocessor:
    """
    Препроцессор данных для датасета Cirrhosis
    Выполняет обработку признаков и кодирование
    """
    
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.cat_features = None
        self.num_features = None
        self.feature_names = None
        self.is_fitted = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def fit(self, df: pd.DataFrame) -> 'CirrhosisDataPreprocessor':
        """
        Обучение препроцессора на тренировочных данных
        
        Args:
            df: тренировочный датафрейм с признаками (без целевой переменной)
        """
        self.logger.info("Обучение препроцессора на тренировочных данных")
        self.logger.debug(f"Входная форма: {df.shape}")
        
        # Определение категориальных и числовых колонок
        self.cat_features = df.select_dtypes(include=["string", "object"]).columns.tolist()
        self.num_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        
        self.logger.info(f"Категориальные признаки: {self.cat_features}")
        self.logger.info(f"Числовые признаки: {self.num_features}")
        
        # Обработка пропущенных значений в категориальных признаках
        if self.cat_features:
            for col in self.cat_features:
                df[col] = df[col].astype(str).fillna('missing')
        
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразование данных с использованием обученного препроцессора
        
        Args:
            df: датафрейм для преобразования
            
        Returns:
            Преобразованный датафрейм
        """
        if not self.is_fitted:
            raise ValueError("Препроцессор должен быть обучен перед преобразованием")
        
        self.logger.debug(f"Преобразование данных с формой: {df.shape}")
        
        # Создание копии для избежания изменений оригинала
        df_transformed = df.copy()
        
        # Обработка категориальных признаков
        if self.cat_features:
            for col in self.cat_features:
                if col in df_transformed.columns:
                    df_transformed[col] = df_transformed[col].astype(str).fillna('missing')
                else:
                    self.logger.warning(f"Колонка {col} не найдена в данных")
        
        # Проверка наличия всех ожидаемых колонок
        expected_cols = self.num_features + self.cat_features
        for col in expected_cols:
            if col not in df_transformed.columns:
                self.logger.error(f"Отсутствует колонка: {col}")
                raise ValueError(f"Колонка {col} не найдена в данных")
        
        return df_transformed[self.num_features + self.cat_features]
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обучение и преобразование"""
        return self.fit(df).transform(df)
    
    def get_feature_names(self) -> list:
        """Получение названий признаков"""
        if self.num_features and self.cat_features:
            return self.num_features + self.cat_features
        elif self.num_features:
            return self.num_features
        elif self.cat_features:
            return self.cat_features
        else:
            return []


class My_Classifier_Model:
    """
    Основной класс классификатора для предсказания выживаемости при циррозе
    """
    
    def __init__(self, model_dir: str = "./model", log_file: str = "./logs/train.log"):
        """
        Инициализация модели
        
        Args:
            model_dir: директория для сохранения/загрузки артефактов модели
            log_file: путь к файлу лога
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройка логгера для экземпляра
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Настройка обработчика файла, если не существует
        if not self.logger.handlers or len(self.logger.handlers) < 2:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            ))
            self.logger.addHandler(file_handler)
        
        self.preprocessor = CirrhosisDataPreprocessor()
        self.model = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.is_trained = False
        
        # Пути для сохранения модели
        self.model_path = self.model_dir / "catboost_model.cbm"
        self.preprocessor_path = self.model_dir / "preprocessor.pkl"
        self.label_encoder_path = self.model_dir / "label_encoder.pkl"
    
    def train(self, dataset_filename: str, optimize: bool = False, n_trials: int = 50) -> Dict[str, Any]:
        """
        Обучение модели на предоставленном датасете
        
        Args:
            dataset_filename: путь к CSV файлу с обучающими данными
            optimize: выполнять ли оптимизацию гиперпараметров с Optuna
            n_trials: количество попыток оптимизации (если optimize=True)
            
        Returns:
            Словарь с результатами обучения
        """
        self.logger.info("=" * 60)
        self.logger.info("НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ")
        self.logger.info("=" * 60)
        
        try:
            # Загрузка данных
            self.logger.info(f"Загрузка датасета из: {dataset_filename}")
            df = pd.read_csv(dataset_filename)
            self.logger.info(f"Форма датасета: {df.shape}")
            self.logger.debug(f"Колонки: {df.columns.tolist()}")
            
            # Проверка наличия целевой колонки
            if 'Status' not in df.columns:
                error_msg = "Датасет должен содержать колонку 'Status' для обучения"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Разделение признаков и целевой переменной
            if 'id' in df.columns:
                df = df.drop(columns=['id'])
            
            y = df['Status'].copy()
            X = df.drop(columns=['Status'])
            
            # Кодирование целевой переменной
            self.logger.info("Кодирование целевой переменной")
            y_encoded = self.label_encoder.fit_transform(y)
            self.classes_ = self.label_encoder.classes_
            self.logger.info(f"Целевые классы: {self.classes_}")
            
            # Обучение препроцессора
            self.logger.info("Обучение препроцессора")
            X_processed = self.preprocessor.fit_transform(X)
            
            # Получение индексов категориальных признаков для CatBoost
            cat_features_indices = []
            if self.preprocessor.cat_features:
                cat_features_indices = [
                    i for i, col in enumerate(X_processed.columns) 
                    if col in self.preprocessor.cat_features
                ]
                self.logger.info(f"Индексы категориальных признаков: {cat_features_indices}")
            
            # Оптимизация гиперпараметров
            if optimize:
                self.logger.info(f"Запуск оптимизации гиперпараметров с {n_trials} попытками")
                best_params = self._optimize_hyperparameters(
                    X_processed, y_encoded, cat_features_indices, n_trials
                )
                self.logger.info(f"Найденные лучшие параметры: {best_params}")
            else:
                # Параметры по умолчанию
                best_params = {
                    'learning_rate': 0.05,
                    'depth': 6,
                    'l2_leaf_reg': 3,
                    'iterations': 1000,
                    'random_seed': 42
                }
            
            # Обучение финальной модели с лучшими параметрами
            self.logger.info("Обучение финальной модели с лучшими параметрами")
            self.model = CatBoostClassifier(
                **best_params,
                loss_function='MultiClass',
                eval_metric='MultiClass',
                cat_features=cat_features_indices if cat_features_indices else None,
                verbose=False,
                early_stopping_rounds=50
            )
            
            # Обучение с валидацией
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed, y_encoded)):
                self.logger.info(f"Обучение фолда {fold + 1}/5")
                
                X_train, X_val = X_processed.iloc[train_idx], X_processed.iloc[val_idx]
                y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
                
                model_fold = CatBoostClassifier(
                    **best_params,
                    loss_function='MultiClass',
                    eval_metric='MultiClass',
                    cat_features=cat_features_indices if cat_features_indices else None,
                    verbose=False,
                    early_stopping_rounds=50
                )
                
                model_fold.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    use_best_model=True
                )
                
                # Предсказание на валидации
                val_pred = model_fold.predict_proba(X_val)
                
                # Вычисление log loss
                from sklearn.metrics import log_loss
                fold_log_loss = log_loss(y_val, val_pred)
                cv_scores.append(fold_log_loss)
                self.logger.info(f"Фолд {fold + 1} log loss: {fold_log_loss:.6f}")
                
                # Сохранение лучшей модели
                if fold == 0 or fold_log_loss < min(cv_scores[:-1]):
                    self.model = model_fold
                    self.logger.debug(f"Обновлена лучшая модель с фолда {fold + 1}")
            
            mean_log_loss = np.mean(cv_scores)
            std_log_loss = np.std(cv_scores)
            self.logger.info(f"CV log loss: {mean_log_loss:.6f} ± {std_log_loss:.6f}")
            
            # Сохранение артефактов
            self._save_artifacts()
            
            self.is_trained = True
            self.logger.info("Обучение модели успешно завершено")
            
            return {
                'status': 'success',
                'cv_log_loss': mean_log_loss,
                'cv_std': std_log_loss,
                'best_params': best_params,
                'classes': self.classes_.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка во время обучения: {str(e)}", exc_info=True)
            raise
    
    def _optimize_hyperparameters(self, X, y, cat_features_indices, n_trials=50) -> Dict[str, Any]:
        """
        Оптимизация гиперпараметров с использованием Optuna
        
        Args:
            X: признаки для обучения
            y: целевая переменная
            cat_features_indices: индексы категориальных признаков
            n_trials: количество попыток оптимизации
            
        Returns:
            Лучшие гиперпараметры
        """
        
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'iterations': trial.suggest_int('iterations', 500, 2000, step=100),
                'random_seed': 42,
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'border_count': trial.suggest_int('border_count', 32, 255),
            }
            
            # Кросс-валидация
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = CatBoostClassifier(
                    **params,
                    loss_function='MultiClass',
                    eval_metric='MultiClass',
                    cat_features=cat_features_indices if cat_features_indices else None,
                    verbose=False,
                    early_stopping_rounds=50
                )
                
                model.fit(
                    X_train, y_train,
                    eval_set=(X_val, y_val),
                    use_best_model=True
                )
                
                val_pred = model.predict_proba(X_val)
                
                from sklearn.metrics import log_loss
                score = log_loss(y_val, val_pred)
                scores.append(score)
            
            return np.mean(scores)
        
        self.logger.info("Запуск оптимизации гиперпараметров Optuna")
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.logger.info(f"Лучшая попытка: {study.best_trial.number}")
        self.logger.info(f"Лучшее значение: {study.best_value:.6f}")
        
        return study.best_params
    
    def predict(self, dataset_filename: str, output_filename: str = "./results/submission.csv") -> pd.DataFrame:
        """
        Выполнение предсказаний на новых данных
        
        Args:
            dataset_filename: путь к CSV файлу для предсказания
            output_filename: путь для сохранения предсказаний
            
        Returns:
            Датафрейм с предсказаниями
        """
        self.logger.info("=" * 60)
        self.logger.info("НАЧАЛО ПРЕДСКАЗАНИЯ")
        self.logger.info("=" * 60)
        
        try:
            # Проверка наличия обученной модели
            if not self.is_trained:
                self.logger.info("Модель не обучена, попытка загрузки с диска")
                if not self.load_model():
                    error_msg = "Обученная модель не найдена. Сначала обучите модель или проверьте наличие файлов модели."
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)
            
            # Загрузка данных
            self.logger.info(f"Загрузка датасета из: {dataset_filename}")
            df = pd.read_csv(dataset_filename)
            self.logger.info(f"Форма датасета: {df.shape}")
            
            # Сохранение ID если присутствуют
            ids = None
            if 'id' in df.columns:
                ids = df['id'].copy()
                X = df.drop(columns=['id'])
            else:
                X = df.copy()
                self.logger.warning("Колонка 'id' не найдена в датасете")
            
            # Преобразование данных
            self.logger.info("Предобработка данных")
            X_processed = self.preprocessor.transform(X)
            
            # Выполнение предсказаний
            self.logger.info("Выполнение предсказаний")
            probabilities = self.model.predict_proba(X_processed)
            
            # Создание датафрейма с предсказаниями
            if ids is not None:
                submission = pd.DataFrame({
                    'id': ids,
                    f'Status_{self.classes_[0]}': probabilities[:, 0],
                    f'Status_{self.classes_[1]}': probabilities[:, 1],
                    f'Status_{self.classes_[2]}': probabilities[:, 2]
                })
            else:
                submission = pd.DataFrame({
                    f'Status_{self.classes_[0]}': probabilities[:, 0],
                    f'Status_{self.classes_[1]}': probabilities[:, 1],
                    f'Status_{self.classes_[2]}': probabilities[:, 2]
                })
            
            # Сохранение предсказаний
            output_path = Path(output_filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            submission.to_csv(output_filename, index=False, float_format='%.6f')
            
            self.logger.info(f"Предсказания сохранены в: {output_filename}")
            self.logger.info(f"Форма предсказаний: {submission.shape}")
            self.logger.debug(f"Первые 5 строк:\n{submission.head()}")
            
            return submission
            
        except Exception as e:
            self.logger.error(f"Ошибка во время предсказания: {str(e)}", exc_info=True)
            raise
    
    def _save_artifacts(self):
        """Сохранение артефактов модели на диск"""
        self.logger.info("Сохранение артефактов модели")
        
        try:
            # Сохранение модели CatBoost
            if self.model:
                self.model.save_model(str(self.model_path))
                self.logger.info(f"Модель сохранена в: {self.model_path}")
            
            # Сохранение препроцессора
            with open(self.preprocessor_path, 'wb') as f:
                pickle.dump(self.preprocessor, f)
            self.logger.info(f"Препроцессор сохранён в: {self.preprocessor_path}")
            
            # Сохранение кодировщика меток
            with open(self.label_encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            self.logger.info(f"Кодировщик меток сохранён в: {self.label_encoder_path}")
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении артефактов: {str(e)}", exc_info=True)
            raise
    
    def load_model(self, model_dir: str = None) -> bool:
        """
        Загрузка артефактов модели с диска
        
        Args:
            model_dir: директория с артефактами модели (по умолчанию: self.model_dir)
            
        Returns:
            True если загрузка успешна, False в противном случае
        """
        if model_dir:
            self.model_dir = Path(model_dir)
            self.model_path = self.model_dir / "catboost_model.cbm"
            self.preprocessor_path = self.model_dir / "preprocessor.pkl"
            self.label_encoder_path = self.model_dir / "label_encoder.pkl"
        
        self.logger.info(f"Загрузка артефактов модели из: {self.model_dir}")
        
        try:
            # Проверка наличия файлов
            if not self.model_path.exists():
                self.logger.error(f"Файл модели не найден: {self.model_path}")
                return False
            
            if not self.preprocessor_path.exists():
                self.logger.error(f"Файл препроцессора не найден: {self.preprocessor_path}")
                return False
            
            if not self.label_encoder_path.exists():
                self.logger.error(f"Файл кодировщика меток не найден: {self.label_encoder_path}")
                return False
            
            # Загрузка модели
            self.model = CatBoostClassifier()
            self.model.load_model(str(self.model_path))
            self.logger.info(f"Модель загружена из: {self.model_path}")
            
            # Загрузка препроцессора
            with open(self.preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            self.logger.info(f"Препроцессор загружен из: {self.preprocessor_path}")
            
            # Загрузка кодировщика меток
            with open(self.label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.logger.info(f"Кодировщик меток загружен из: {self.label_encoder_path}")
            
            # Установка классов
            self.classes_ = self.label_encoder.classes_
            self.is_trained = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке артефактов: {str(e)}", exc_info=True)
            return False


def main():
    """Основная точка входа CLI"""
    parser = argparse.ArgumentParser(description='Предсказание выживаемости пациентов с циррозом')
    subparsers = parser.add_subparsers(dest='command', help='Команды')
    
    # Команда для обучения
    train_parser = subparsers.add_parser('train', help='Обучение модели')
    train_parser.add_argument('--dataset', type=str, required=True, 
                              help='Путь к CSV файлу с обучающими данными')
    train_parser.add_argument('--optimize', action='store_true', 
                              help='Выполнить оптимизацию гиперпараметров с Optuna')
    train_parser.add_argument('--trials', type=int, default=50,
                              help='Количество попыток оптимизации (по умолчанию: 50)')
    train_parser.add_argument('--model-dir', type=str, default='./model',
                              help='Директория для сохранения артефактов модели')
    
    # Команда для предсказания
    predict_parser = subparsers.add_parser('predict', help='Выполнение предсказаний')
    predict_parser.add_argument('--dataset', type=str, required=True,
                                help='Путь к CSV файлу для предсказания')
    predict_parser.add_argument('--output', type=str, default='./results/submission.csv',
                                help='Путь для сохранения предсказаний CSV')
    predict_parser.add_argument('--model-dir', type=str, default='./model',
                                help='Директория с артефактами модели')
    
    # Парсинг аргументов
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Настройка логирования
    log_dir = Path("./logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{args.command}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Настройка глобального логгера
    global logger
    logger = setup_logger('cirrhosis_model', log_file)
    
    # Выполнение команды
    if args.command == 'train':
        logger.info(f"Обучение модели с датасетом: {args.dataset}")
        model = My_Classifier_Model(model_dir=args.model_dir, log_file=log_file)
        results = model.train(
            dataset_filename=args.dataset,
            optimize=args.optimize,
            n_trials=args.trials
        )
        logger.info(f"Результаты обучения: {results}")
        
    elif args.command == 'predict':
        logger.info(f"Выполнение предсказаний с датасетом: {args.dataset}")
        model = My_Classifier_Model(model_dir=args.model_dir, log_file=log_file)
        if model.load_model():
            predictions = model.predict(
                dataset_filename=args.dataset,
                output_filename=args.output
            )
            logger.info(f"Предсказания сохранены в: {args.output}")
        else:
            logger.error("Не удалось загрузить модель. Сначала обучите модель или проверьте директорию с моделью.")
            sys.exit(1)


if __name__ == '__main__':
    main()