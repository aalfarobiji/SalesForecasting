# 📈 Sales Forecasting — ARIMA vs LSTM

A time series forecasting project that compares classical statistical methods (ARIMA) with deep learning (LSTM) for predicting daily retail sales at Store #5.

---

## 📌 Overview

This project forecasts daily sales using two approaches:

- **ARIMA** — AutoRegressive Integrated Moving Average (via `pmdarima` auto-ARIMA)
- **LSTM** — Long Short-Term Memory neural network (via Keras/TensorFlow)

The goal is to evaluate and compare the predictive performance of both models on real retail time series data.

---

## 📂 Dataset

**File:** `store5.csv`

Data covers **Store #5** from **January 2013 to August 2017** with **55,572 rows** across 33 product families.

| Column | Description |
|---|---|
| `id` | Unique record identifier |
| `date` | Transaction date |
| `store_nbr` | Store number (fixed at 5) |
| `family` | Product category (33 unique families) |
| `sales` | Total sales amount |
| `onpromotion` | Number of items on promotion |
| `dcoilwtico` | Daily WTI crude oil price |

> The dataset is aggregated into **daily total sales** (1,684 time steps) before modeling.

---

## 🔧 Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `statsmodels` | Time series decomposition & ARIMA |
| `pmdarima` | Auto-ARIMA model selection |
| `scikit-learn` | Preprocessing (MinMaxScaler) & metrics |
| `keras` / `tensorflow` | LSTM model (Sequential, Bidirectional, GRU, BatchNorm) |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/aalfarobiji/sales-forecasting.git
cd sales-forecasting
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn statsmodels pmdarima scikit-learn tensorflow keras
```

### 3. Add the dataset

Place `store5.csv` in the root directory of the project.

### 4. Run the notebook

```bash
jupyter notebook Sales_Forecasting.ipynb
```

Or open directly in [Google Colab](https://colab.research.google.com/).

---

## 🧪 Workflow

```
Raw Data (store5.csv)
    │
    ▼
Data Loading & Exploration
    │
    ▼
Preprocessing
  ├── Date parsing & indexing
  ├── Daily sales aggregation
  └── Missing value handling (oil price)
    │
    ▼
Exploratory Data Analysis
  ├── Time series plot
  ├── Autocorrelation (ACF/PACF)
  └── Seasonal decomposition
    │
    ├──────────────────────────────┐
    ▼                              ▼
ARIMA Model                   LSTM Model
  ├── Auto-ARIMA (pmdarima)     ├── MinMax Scaling
  ├── Stationarity check        ├── Sequence windowing
  └── Forecasting               ├── Bidirectional LSTM layers
                                ├── Dropout + BatchNorm
                                ├── EarlyStopping
                                └── Inverse transform predictions
    │                              │
    └──────────────┬───────────────┘
                   ▼
         Model Comparison & Evaluation
```

---

## 📊 Results

| Metric | ARIMA | LSTM |
|---|---|---|
| **R²** | -0.358 | -0.123 |
| **MSE** | 2,940,283 | 2,373,588 |
| **RMSE** | 1,714.73 | **1,540.65** ✅ |
| **MAE** | 1,298.39 | **1,187.08** ✅ |
| **MAPE (%)** | 12.03% | **11.40%** ✅ |
| **MASE** | 0.846 | **0.774** ✅ |

> **LSTM outperforms ARIMA** across all metrics, achieving lower error and better generalization on the test set.

---

## 📁 Project Structure

```
sales-forecasting/
│
├── Sales_Forecasting.ipynb   # Main notebook
├── store5.csv                # Dataset (not included — add manually)
└── README.md
```

---

## 💡 Key Insights

- Sales data from Store #5 exhibits clear **weekly seasonality** and occasional anomalies (e.g., zero sales on holidays).
- ARIMA struggles with the non-linear patterns in the data, evidenced by a negative R² score.
- LSTM captures temporal dependencies better but would benefit from further hyperparameter tuning and more training data.
- MAPE of ~11.4% (LSTM) suggests reasonable forecast accuracy for a retail setting.

---

## 🔮 Potential Improvements

- Incorporate `onpromotion` and oil price (`dcoilwtico`) as exogenous features
- Try **Prophet** or **N-BEATS** for comparison
- Experiment with **multivariate LSTM** using all product families
- Extend to all stores for a generalized model
- Hyperparameter tuning with grid search or Optuna

---

## 📄 License

This project is for educational purposes. Feel free to fork and experiment!
