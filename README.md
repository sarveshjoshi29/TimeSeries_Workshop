# Time Series & Order Book Workshop

Code and slides for the IITG.AI workshop on forecasting volatility from limit order book data.

We use the [Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction) dataset. For each stock and 10-minute window (`time_id`), the pipeline reads the book and trade updates and predicts the realized volatility of the **next** 10 minutes.

---

## Repository Structure

```text
timeseries_workshop/
├── pipeline.py                          # End-to-end pipeline: load, clean, features, models, report
├── requirements.txt                     # Python dependencies
├── timeseries_orderbook_workshop.pptx   # Workshop slides
└── .gitignore
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/sarveshjoshi29/timeseries_workshop.git
cd timeseries_workshop
```

### 2. Create a virtual environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Get the data

Download the dataset from the [Kaggle competition page](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data) (you need to accept the competition rules first). Put these files in the repository root:

```text
book_train.parquet.zip
trade_train.parquet.zip
train.csv.zip
```

The pipeline extracts only the stocks it needs from these zips, so you do not have to unzip them yourself. The data files are git-ignored.

---

## What the Pipeline Does

1. **Load**: read book, trade and target data for the chosen stocks.
2. **Clean**: drop nulls, non-positive prices or sizes, crossed books, out-of-order levels and duplicate rows.
3. **Densify**: turn the event stream into a 1-second grid using forward-fill only (no look-ahead).
4. **Features**: WAP, log returns, realized vol over the window and its last 60s/300s, EWMA vol, spread, imbalance, trade stats, and a sequence of 10-second realized vols.
5. **Split**: train/test split grouped by `time_id`, so no market moment appears in both sets.
6. **Models**:
   - Naive (next RV = this window's RV)
   - Random Forest
   - XGBoost
   - GARCH(1,1), one fit per window
   - Hybrid: GARCH + XGBoost residual
   - NN alone (LSTM + MLP)
   - Hybrid: GARCH + NN residual
7. **Report**: RMSPE (the competition metric) and R² for every model on the same held-out windows.

---

## Running the Pipeline

### Default run (15 workshop stocks, all models)
```bash
python pipeline.py
```
This runs on stocks 5, 18, 23, 30, 33, 36, 43, 50, 53, 59, 69, 76, 105, 116 and 119. These are the stocks behind the results in the slides.

### Skip the neural networks
`--no-nn` skips the two neural network models (NN alone, and GARCH + NN). All other models still run, including GARCH + XGBoost.
```bash
python pipeline.py --no-nn
```

### Choose your own stocks
```bash
python pipeline.py --stocks 1,2,3 --no-nn
```

### All 112 stocks
```bash
python pipeline.py --stocks all --no-nn
```
`--stocks all` reads the stock list from the `book_train.parquet/` folder, so unzip `book_train.parquet.zip` and `trade_train.parquet.zip` fully before you use it. This run takes much longer and needs much more memory than the default run.

### Interactive debugging
Add `-i` to keep the DataFrames and fitted models in memory after the run:
```bash
python -i pipeline.py --no-nn
```

---

## Outputs

| File | Contents |
|---|---|
| `results.csv` | RMSPE and R² for each model on the held-out test windows |
| `results_per_stock.csv` | RMSPE for each model, per stock |
| `features_cache/stock_<id>.parquet` | Cached features and GARCH forecasts for each stock. Later runs reuse them. Delete this folder to rebuild. |

### Results on the 15 workshop stocks (11,490 test windows)

| Model | RMSPE ↓ | R² |
|---|---|---|
| Naive | 0.321 | 0.68 |
| GARCH(1,1) | 0.288 | 0.71 |
| Random Forest | 0.239 | 0.78 |
| XGBoost | 0.237 | 0.74 |
| GARCH + XGBoost | 0.234 | 0.78 |
| GARCH + NN | 0.234 | 0.78 |

---

## Workshop Materials

Open `timeseries_orderbook_workshop.pptx` in PowerPoint, Keynote or Google Slides. The deck covers order book basics, data cleaning, avoiding look-ahead, feature engineering, tree models, GARCH and the hybrid GARCH + NN model.
