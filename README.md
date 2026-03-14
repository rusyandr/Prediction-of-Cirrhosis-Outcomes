# Prediction-of-Cirrhosis-Outcomes

**Студенты:** Чéрникова Ангелина и Агафонов Руслан   
**Группа:** 972401

## Описание проекта

Предсказание выживаемости пациентов с циррозом печени (C, CL, D) с использованием CatBoost и MLOps практик.

**Результат:** Kaggle Log Loss **0.34601**

## Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/rusyandr/Prediction-of-Cirrhosis-Outcomes.git
cd Prediction-of-Cirrhosis-Outcomes
```

### 2. Установка Poetry

#### Windows (PowerShell):
```powershell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

#### Linux/Mac:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### Проверка установки:
```bash
poetry --version
```

### 3. Установка зависимостей проекта

```bash
# Установить все библиотеки из pyproject.toml
poetry install
```

Будут установлены:
- pandas, numpy - обработка данных
- scikit-learn - метрики и preprocessing
- catboost - основная модель
- optuna - оптимизация гиперпараметров
- matplotlib, seaborn - визуализация
- jupyter - для нотбуков
- lightgbm - для ансамблей

### 4. Активация виртуального окружения

```bash
# Вариант 1: Активировать окружение
poetry shell

# Вариант 2: Использовать poetry run (для отдельных команд)
poetry run python src/model.py --help
```

### 5. Скачивание данных

1. Перейдите на [Kaggle Competition](https://www.kaggle.com/competitions/playground-series-s3e26)
2. Скачайте файлы:
   - `train.csv`
   - `test.csv`
   - `sample_submission.csv`
3. Положите их в папку `data/`:

```bash
# Создать папку data если её нет
mkdir -p data
# Переместите скачанные файлы в папку data
```

## Обучение модели

### Базовое обучение (без оптимизации)
```bash
poetry run python src/model.py train --dataset=./data/train.csv
```

### Обучение с оптимизацией гиперпараметров (Optuna)
```bash
# 50 попыток оптимизации
poetry run python src/model.py train --dataset=./data/train.csv --optimize --trials=50
```

### Что происходит при обучении:
- Автоматическая обработка пропущенных значений
- Кодирование категориальных признаков
- 5-кратная кросс-валидация
- Логирование в папку `logs/`
- Сохранение модели в папку `model/`

## Предсказание

```bash
# Сделать предсказания для тестовых данных
poetry run python src/model.py predict --dataset=./data/test.csv --output=./results/submission.csv
```

Результат сохранится в `results/submission.csv` в формате:
```
id,Status_C,Status_CL,Status_D
15000,0.942032,0.028616,0.029351
...
```

## Jupyter Notebook (EDA и Optuna)

### Запуск Jupyter
```bash
# Убедитесь, что зависимости установлены
poetry install

# Запустить Jupyter
poetry run jupyter notebook
```

### В браузере откроется http://localhost:8888
1. Перейдите в папку `notebooks/`
2. Откройте `cirrhosis_analysis.ipynb`
3. Выполните все ячейки (Kernel → Restart & Run All)

**Нотбук содержит:**
- Разведочный анализ данных (EDA)
- Визуализацию распределений
- Baseline модель
- Оптимизацию гиперпараметров Optuna
- Сравнение моделей

## Создание .whl пакета

```bash
# Собрать пакет
poetry build

# Файлы появятся в папке dist/
ls dist/
# cirrhosis_prediction-0.1.0-py3-none-any.whl
# cirrhosis_prediction-0.1.0.tar.gz

# Установка пакета
pip install dist\prediction_of_cirrhosis_outcomes-0.1.0-py3-none-any.whl
```

## Структура проекта

```
Prediction-of-Cirrhosis-Outcomes/
│
├── data/                          # Данные (игнорируются git)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── src/                            # Исходный код
│   └── model.py                    # Основной класс модели и CLI
│
├── model/                          # Сохраненные артефакты модели
│   ├── catboost_model.cbm
│   ├── preprocessor.pkl
│   └── label_encoder.pkl
│
├── results/                        # Результаты предсказаний
│   └── submission.csv
│
├── notebooks/                      # Jupyter нотбуки
│   └── cirrhosis_analysis.ipynb
│
├── logs/                           # Логи обучения
│   └── train_*.log
│
├── dist/                           # Собранные .whl пакеты
│   ├── cirrhosis_prediction-*.whl
│   └── cirrhosis_prediction-*.tar.gz
│
├── .gitignore
├── README.md
├── pyproject.toml                  # Конфигурация Poetry
└── poetry.lock                     # Зафиксированные версии
```

##  Результаты

|      Метрика      |    Значение    |
|-------------------|----------------|
| Kaggle Log Loss   | **0.34601**    |
| CV Log Loss       | 0.377 ± 0.0126 |
| Baseline Log Loss | 0.3956         |
| Улучшение         | 0.0496         |
