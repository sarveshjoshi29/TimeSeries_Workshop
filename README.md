# Time Series & Order Book Workshop

A hands-on workshop repository for analyzing and modeling financial time series and limit order book (LOB) data.

---

## Repository Structure

```text
TimeSeries_Workshop/
├── pipeline.py                          # Data ingestion, feature engineering, and model training pipeline
├── requirements.txt                     # Python dependencies
├── timeseries_orderbook_workshop.pptx   # Workshop presentation slides
└── .gitignore                           # Git ignore rules
```

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sarveshjoshi29/timeseries_workshop.git
cd timeseries_workshop
```

### 2. Set Up a Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Pipeline

### Fast Run (All Stocks, No GARCH + Neural Network and GARCH + XGB hybrids)
Process all tickers while bypassing neural network training to iterate and verify features quickly:

```bash
python pipeline.py --stocks all --no-nn
```

### Full Run (All Stocks with GARCH + Neural Network and GARCH + XGB hybrids)
Run the complete pipeline including neural network models across the entire stock dataset:

```bash
python pipeline.py --stocks all
```

### Selective Stocks Run
Run the pipeline for an individual stock without neural networks:

```bash
python pipeline.py --stocks 1,2,3 --no-nn
```

### Interactive Debugging
Add Python's interactive flag `-i` to keep processed DataFrames, order book snapshots, and pipeline objects loaded in memory after execution:

```bash
python -i pipeline.py --stocks all --no-nn
```

---

## Workshop Materials

Open `timeseries_orderbook_workshop.pptx` in Microsoft PowerPoint, Apple Keynote, or Google Slides to follow along with the workshop deck, which details order book microstructure, feature design, and model architectures.