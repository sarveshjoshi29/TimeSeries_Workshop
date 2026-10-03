"""
Optiver Realized Volatility — full pipeline.

Task: for each (stock_id, time_id), use the 10 minutes of book and trade
data (seconds_in_bucket 0-599) to predict `target` from train.csv: the
realized volatility of the NEXT 10 minutes, which are not in the files.

Stages:
  1. load       — extract chosen stock partitions, read book/trade/train
  2. clean      — sanity checks, drop bad rows
  3. densify    — event stream -> 1-second grid (forward-fill only)
  4. features   — WAP, log returns, spread, imbalance, trade stats,
                  10-second bucket realized vol sequence
  5. split      — train/test grouped by time_id (no window in both)
  6. models     — naive, RandomForest, XGBoost, GARCH,
                  NN alone, GARCH+NN hybrid
  7. report     — RMSPE / R2 on the same held-out test windows

  
TO RUN :- 
1. Create a virtual env -> python -m venv .venv
2. Install required libraries -> pip install -r requirements.txt
3. Run this file 

Run: python pipeline.py                     # runs only the 15 stocks

    args: 1) --stocks
    Eg :  Run python pipeline.py --stocks 1,2,3
    mention the ids of stocks separated by comma OR use all OR no args for running the 15 stocks only
    2) -- no-nn
    Eg: python pipeline.py --stocks all --no-nn 
    Does not run the residual GARCH + NN and GARCH + XGBoost models
"""
import argparse
import os
import time
import warnings
import zipfile

# per-window GARCH fits on short noisy series emit thousands of
# convergence warnings; failures are handled by explicit fallbacks below
os.environ.setdefault("PYTHONWARNINGS", "ignore")   # inherited by joblib workers
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from arch import arch_model
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

# 15 stocks spread evenly across the volatility range (by mean target)
STOCK_IDS = [5, 18, 23, 30, 33, 36, 43, 50, 53, 59, 69, 76, 105, 116, 119]
WINDOW = 600          # seconds of book/trade data per window
BUCKET = 10           # seconds per realized-vol bucket
N_BUCKETS = WINDOW // BUCKET
SEED = 0
KEYS = ["stock_id", "time_id"]
BOOK_COLS = ["bid_price1", "ask_price1", "bid_price2", "ask_price2",
             "bid_size1", "ask_size1", "bid_size2", "ask_size2"]
STATIC_FEATURES = [
    "rv_window", "rv_last_60s", "rv_last_300s", "ewma_vol_30s", "ewma_vol_120s",
    "wap_std", "spread_mean", "spread_max",
    "imbalance_mean", "imbalance_std", "n_updates",
    "trade_count", "trade_vol", "trade_price_std",
]
BUCKET_COLS = [f"rv_b{i:02d}" for i in range(N_BUCKETS)]
CACHE_DIR = "features_cache"


# ---------------------------------------------------------------- 1. load

def extract_stocks(stock_ids=STOCK_IDS):
    for zname, out in [("book_train.parquet.zip", "book_train.parquet"),
                       ("trade_train.parquet.zip", "trade_train.parquet")]:
        with zipfile.ZipFile(zname) as z:
            for member in z.namelist():
                sid = member.split("/")[0].removeprefix("stock_id=")
                if sid.isdigit() and int(sid) in stock_ids and not os.path.exists(os.path.join(out, member)):
                    z.extract(member, out)
    if not os.path.exists("train.csv"):
        with zipfile.ZipFile("train.csv.zip") as z:
            z.extract("train.csv", ".")


def load(stock_ids=STOCK_IDS):
    # filter at read time: the full book is 167M rows across 112 stocks
    only = [("stock_id", "in", list(stock_ids))]
    book = pd.read_parquet("book_train.parquet", filters=only)
    trade = pd.read_parquet("trade_train.parquet", filters=only)
    target = pd.read_csv("train.csv")
    # the partition column comes back as a category; make it a plain int
    for df in (book, trade):
        df["stock_id"] = df["stock_id"].astype(int)
    target = target[target.stock_id.isin(stock_ids)]
    return book, trade, target


# --------------------------------------------------------------- 2. clean

def clean_book(book):
    checks = {
        "null values": book[BOOK_COLS].isna().any(axis=1),
        "non-positive price": (book[["bid_price1", "ask_price1", "bid_price2", "ask_price2"]] <= 0).any(axis=1),
        "non-positive size": (book[["bid_size1", "ask_size1", "bid_size2", "ask_size2"]] <= 0).any(axis=1),
        "crossed/locked book (bid1 >= ask1)": book.bid_price1 >= book.ask_price1,
        "levels out of order": (book.bid_price2 > book.bid_price1) | (book.ask_price2 < book.ask_price1),
        "duplicate (stock, time, second)": book.duplicated(KEYS + ["seconds_in_bucket"]),
    }
    return _drop(book, checks, "book")


def _drop(df, checks, label):
    """Drop rows failing any check; print one line naming the checks that fired."""
    bad = np.zeros(len(df), dtype=bool)
    fired = []
    for name, mask in checks.items():
        n = int(mask.sum())
        if n:
            fired.append(f"{name}: {n:,}")
        bad |= mask.to_numpy()
    print(f"  {label}: {len(df):,} rows, dropped {int(bad.sum()):,}" + (f" ({'; '.join(fired)})" if fired else ""))
    return df[~bad].sort_values(KEYS + ["seconds_in_bucket"]).reset_index(drop=True)


def clean_trade(trade):
    checks = {
        "null values": trade[["price", "size", "order_count"]].isna().any(axis=1),
        "non-positive price/size/count": (trade[["price", "size", "order_count"]] <= 0).any(axis=1),
        "duplicate (stock, time, second)": trade.duplicated(KEYS + ["seconds_in_bucket"]),
    }
    return _drop(trade, checks, "trade")


# ------------------------------------------------------------- 3. densify

def densify(book):
    """Event stream -> one row per second 0..599, forward-filled.

    Forward-fill only: the book at second t is the last update at or
    before t. Back-filling would copy a *later* update into earlier
    seconds — lookahead inside the window — so seconds before a window's
    first update are dropped instead.
    """
    windows = book[KEYS].drop_duplicates()
    grid = windows.loc[windows.index.repeat(WINDOW)].reset_index(drop=True)
    grid["seconds_in_bucket"] = np.tile(np.arange(WINDOW), len(windows))
    dense = grid.merge(book, on=KEYS + ["seconds_in_bucket"], how="left")
    dense["is_update"] = dense["bid_price1"].notna()
    dense[BOOK_COLS] = dense.groupby(KEYS)[BOOK_COLS].ffill()
    return dense.dropna(subset=["bid_price1"]).reset_index(drop=True)


# ------------------------------------------------------------ 4. features

def add_book_columns(dense):
    size_sum = dense.bid_size1 + dense.ask_size1
    dense["wap"] = (dense.bid_price1 * dense.ask_size1 + dense.ask_price1 * dense.bid_size1) / size_sum
    dense["spread"] = dense.ask_price1 - dense.bid_price1
    dense["imbalance"] = (dense.bid_size1 - dense.ask_size1) / size_sum
    dense["log_ret"] = dense.groupby(KEYS)["wap"].transform(lambda s: np.log(s).diff()).fillna(0.0)
    assert np.isfinite(dense[["wap", "imbalance", "log_ret"]].to_numpy()).all()
    return dense


def build_features(dense, trade, target):
    dense["ret_sq"] = dense.log_ret ** 2
    g = dense.groupby(KEYS)
    feat = g.agg(
        rv_window=("ret_sq", "sum"),
        wap_std=("wap", "std"),
        spread_mean=("spread", "mean"), spread_max=("spread", "max"),
        imbalance_mean=("imbalance", "mean"), imbalance_std=("imbalance", "std"),
        n_updates=("is_update", "sum"),
    )
    feat["rv_window"] = np.sqrt(feat.rv_window)
    # lags: realized vol of the most recent stretch of the window
    for secs in (60, 300):
        recent = dense[dense.seconds_in_bucket >= WINDOW - secs]
        feat[f"rv_last_{secs}s"] = np.sqrt(recent.groupby(KEYS)["ret_sq"].sum())
    # EWMA of squared returns, read at the window's last second. adjust=False is
    # the causal recursion: each value uses only that second and earlier ones.
    for halflife in (30, 120):
        ewma = dense.groupby(KEYS)["ret_sq"].ewm(halflife=halflife, adjust=False).mean()
        feat[f"ewma_vol_{halflife}s"] = np.sqrt(ewma.groupby(level=[0, 1]).last())

    # 10-second bucket realized vol: the sequence the LSTM reads
    dense["bucket"] = dense.seconds_in_bucket // BUCKET
    buckets = np.sqrt(dense.groupby(KEYS + ["bucket"])["ret_sq"].sum()).unstack("bucket")
    buckets = buckets.reindex(columns=range(N_BUCKETS)).fillna(0.0)   # no update in bucket = no price move
    buckets.columns = BUCKET_COLS

    tfeat = trade.groupby(KEYS).agg(
        trade_count=("price", "size"), trade_vol=("size", "sum"), trade_price_std=("price", "std"),
    )

    df = feat.join(buckets).join(tfeat).reset_index()
    df[["trade_count", "trade_vol", "trade_price_std"]] = df[["trade_count", "trade_vol", "trade_price_std"]].fillna(0.0)
    lag_cols = ["wap_std", "imbalance_std", "rv_last_60s", "rv_last_300s", "ewma_vol_30s", "ewma_vol_120s"]
    df[lag_cols] = df[lag_cols].fillna(0.0)
    df = df.merge(target, on=KEYS, how="inner")

    before = len(df)
    df = df[(df.target > 0) & (df.n_updates >= 10)].reset_index(drop=True)
    print(f"  windows: {before:,} -> {len(df):,} after dropping zero targets / <10 book updates")
    return df


# ---------------------------------------------------------------- 5. split

def split(df):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(df, groups=df.time_id))
    train, test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    # same time_id = same market moment across stocks: keep it on one side only
    assert not set(train.time_id) & set(test.time_id)
    return train, test


# -------------------------------------------------------- 6. stats models

GARCH_AGG = 5   # seconds per GARCH return: 1-second returns are mostly zeros


def garch_one(returns, rv):
    """GARCH(1,1) on 5-second returns (bps), variance-targeted forecast of the next 600s.

    GARCH supplies the persistence (alpha + beta) and the next-step
    variance; the long-run level it decays toward is pinned to this
    window's own average squared return ("variance targeting"). Letting
    GARCH also estimate the long-run level from 120 points is what
    makes raw per-window forecasts explode.
    """
    x = returns[: len(returns) // GARCH_AGG * GARCH_AGG].reshape(-1, GARCH_AGG).sum(1) * 1e4
    horizon = WINDOW // GARCH_AGG
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = arch_model(x, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")
            next_var = res.forecast(horizon=1, reindex=False).variance.values[-1][0]
        persistence = min(res.params["alpha[1]"] + res.params["beta[1]"], 0.999)
        long_run = np.mean(x ** 2)
        f = long_run + persistence ** np.arange(horizon) * (next_var - long_run)
        pred = np.sqrt(np.clip(f, 0, None).sum() / 1e8)
        if not np.isfinite(pred) or pred <= 0:
            raise ValueError
        return pred
    except Exception:
        return rv


def stock_table(sid, target):
    """Stages 1-4 plus the GARCH forecast for one stock; one row per window."""
    book, trade, _ = load([sid])
    book, trade = clean_book(book), clean_trade(trade)
    dense = add_book_columns(densify(book))
    df = build_features(dense, trade, target[target.stock_id == sid])
    rets = dense.groupby(KEYS)["log_ret"].apply(np.asarray)
    df["garch_pred"] = Parallel(n_jobs=-1, batch_size=32)(
        delayed(garch_one)(rets.loc[(s, t)], v) for s, t, v in zip(df.stock_id, df.time_id, df.rv_window))
    return df


def all_tables(stock_ids):
    os.makedirs(CACHE_DIR, exist_ok=True)
    target = pd.read_csv("train.csv")
    tables = []
    for i, sid in enumerate(stock_ids, 1):
        path = os.path.join(CACHE_DIR, f"stock_{sid}.parquet")
        if os.path.exists(path):
            tables.append(pd.read_parquet(path))
            continue
        t0 = time.time()
        print(f"[{i}/{len(stock_ids)}] stock {sid}", flush=True)
        df = stock_table(sid, target)
        df.to_parquet(path)
        tables.append(df)
        print(f"  done in {time.time() - t0:.0f}s", flush=True)
    return pd.concat(tables, ignore_index=True)


# ----------------------------------------------------- 6b. hybrid GARCH+NN

class ResidualNet(torch.nn.Module):
    """LSTM over the 60-bucket vol sequence + MLP over static features.

    Output is a residual in log space: prediction = baseline * exp(out).
    With the GARCH forecast as baseline this is the hybrid; with a
    constant baseline it is the "NN alone" comparison.
    """
    def __init__(self, n_static, hidden=16):
        super().__init__()
        self.lstm = torch.nn.LSTM(1, hidden, batch_first=True)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden + n_static, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))

    def forward(self, seq, static):
        _, (h, _) = self.lstm(seq.unsqueeze(-1))
        return self.head(torch.cat([h[-1], static], dim=1)).squeeze(1)


def nn_inputs(df, stats):
    seq = np.log(df[BUCKET_COLS].to_numpy() + 1e-5)
    static = np.column_stack([
        np.log(df[["rv_window", "garch_pred"]].to_numpy() + 1e-6),
        df[STATIC_FEATURES[2:]].to_numpy(),
    ])
    if stats is None:     # fit scalers on TRAIN only
        stats = (seq.mean(), seq.std(), static.mean(0), static.std(0) + 1e-9)
    seq = (seq - stats[0]) / stats[1]
    static = (static - stats[2]) / stats[3]
    return torch.tensor(seq, dtype=torch.float32), torch.tensor(static, dtype=torch.float32), stats


def rmspe_loss(pred, y):
    return torch.sqrt(torch.mean(((pred - y) / y) ** 2))


def train_residual_nn(train, test, baseline_col):
    torch.manual_seed(SEED)
    # validation for early stopping: split TRAIN by time_id again, never touch test
    fit, val = split(train)
    seq_f, st_f, stats = nn_inputs(fit, None)
    seq_v, st_v, _ = nn_inputs(val, stats)
    seq_t, st_t, _ = nn_inputs(test, stats)

    def base(d):
        if baseline_col is None:
            return torch.full((len(d),), float(np.exp(np.log(fit.target).mean())))
        return torch.tensor(d[baseline_col].to_numpy(), dtype=torch.float32)

    b_f, b_v, b_t = base(fit), base(val), base(test)
    y_f = torch.tensor(fit.target.to_numpy(), dtype=torch.float32)
    y_v = torch.tensor(val.target.to_numpy(), dtype=torch.float32)

    net = ResidualNet(st_f.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, best_state, patience = np.inf, None, 0
    for epoch in range(100):
        net.train()
        for idx in torch.randperm(len(y_f)).split(256):
            opt.zero_grad()
            loss = rmspe_loss(b_f[idx] * torch.exp(net(seq_f[idx], st_f[idx])), y_f[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = rmspe_loss(b_v * torch.exp(net(seq_v, st_v)), y_v).item()
        if v < best - 1e-4:
            best, best_state, patience = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 8:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        return (b_t * torch.exp(net(seq_t, st_t))).numpy(), epoch + 1


# ------------------------------------------------------------- 7. report

def rmspe(y, p):
    return float(np.sqrt(np.mean(((y - p) / y) ** 2)))


def fit_rmspe_xgb(train, cols, label, name):
    """XGBoost trained and tuned on RMSPE of `label`.

    Weights 1/label^2 make the squared-error loss equal squared percentage
    error. For the hybrid, label = target / garch and RMSPE of that ratio
    equals RMSPE of the final prediction garch * ratio, so the same metric
    drives early stopping. Grid and early stopping use a validation split
    of TRAIN grouped by time_id; the winner is refit on all of TRAIN.
    """
    data = train.assign(_label=label)
    fit, val = split(data)
    grid = [dict(max_depth=d, learning_rate=lr, min_child_weight=mcw)
            for d in (3, 4, 6) for lr in (0.03, 0.1) for mcw in (1, 50)]
    best = None
    for params in grid:
        m = XGBRegressor(n_estimators=3000, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         early_stopping_rounds=100, eval_metric=rmspe, random_state=SEED, **params)
        m.fit(fit[cols], fit._label, sample_weight=1 / fit._label ** 2,
              eval_set=[(val[cols], val._label)], verbose=False)
        if best is None or m.best_score < best[0]:
            best = (m.best_score, params, m.best_iteration + 1)
    score, params, n_trees = best
    print(f"  {name}: best of {len(grid)} = {params}, {n_trees} trees, validation RMSPE {score:.4f}")
    final = XGBRegressor(n_estimators=n_trees, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                         random_state=SEED, **params)
    return final.fit(train[cols], label, sample_weight=1 / label ** 2)


def main(stock_ids, with_nn):
    extract_stocks(stock_ids)
    print("== clean + densify + features + GARCH, per stock ==")
    df = all_tables(stock_ids)

    train, test = split(df)
    print(f"train windows: {len(train):,}   test windows: {len(test):,}   "
          f"(distinct time_ids: {train.time_id.nunique():,} / {test.time_id.nunique():,})")

    y_tr, y_te = train.target.to_numpy(), test.target.to_numpy()
    results = {}
    results["Naive: next RV = this RV"] = test.rv_window.to_numpy()

    print("\n== RandomForest / XGBoost ==")
    w = 1 / y_tr ** 2         # RMSPE weights each window by 1/target^2
    rf = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=20,
                               random_state=SEED, n_jobs=-1)
    rf.fit(train[STATIC_FEATURES], y_tr, sample_weight=w)
    results["RandomForest"] = rf.predict(test[STATIC_FEATURES])

    xgb = fit_rmspe_xgb(train, STATIC_FEATURES, y_tr, "XGBoost")
    results["XGBoost"] = xgb.predict(test[STATIC_FEATURES])

    results["GARCH(1,1), variance-targeted"] = test.garch_pred.to_numpy()

    # residual fitting: GARCH is the baseline, XGBoost learns the ratio it missed
    # (target / garch). An unweighted log-ratio fit scored 0.252 — the loss has to
    # be the metric.
    hybrid_cols = STATIC_FEATURES + ["garch_pred"]
    hx = fit_rmspe_xgb(train, hybrid_cols, y_tr / train.garch_pred.to_numpy(), "GARCH + XGBoost")
    results["Hybrid: GARCH + XGBoost residual"] = test.garch_pred.to_numpy() * hx.predict(test[hybrid_cols])

    if with_nn:
        print("== neural nets ==")
        results["NN alone (LSTM+MLP)"], ep1 = train_residual_nn(train, test, None)
        results["Hybrid: GARCH + NN residual"], ep2 = train_residual_nn(train, test, "garch_pred")
        print(f"early-stopped at epoch {ep1} (NN alone), {ep2} (hybrid)")

    table = pd.DataFrame(
        [(name, rmspe(y_te, p), r2_score(y_te, p)) for name, p in results.items()],
        columns=["model", "RMSPE", "R2"],
    )
    print("\n== results (same held-out test windows) ==")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    table.to_csv("results.csv", index=False)

    imp = pd.Series(xgb.feature_importances_, index=STATIC_FEATURES).sort_values(ascending=False)
    print("\nXGBoost feature importance:")
    print(imp.to_string(float_format=lambda x: f"{x:.3f}"))

    shown = [m for m in results if m != "Naive: next RV = this RV"]
    per_stock = test.assign(**results).groupby("stock_id").apply(
        lambda g: pd.Series({m: rmspe(g.target.to_numpy(), g[m].to_numpy()) for m in shown}),
        include_groups=False)
    per_stock.to_csv("results_per_stock.csv")
    many = len(per_stock) > 10
    print("\nRMSPE per stock" + (" (5 easiest / 5 hardest for XGBoost):" if many else ":"))
    if many:
        per_stock = per_stock.sort_values("XGBoost").iloc[list(range(5)) + list(range(-5, 0))]
    print(per_stock.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", default=",".join(map(str, STOCK_IDS)),
                    help='comma-separated stock ids, or "all"')
    ap.add_argument("--no-nn", action="store_true", help="skip the NN and hybrid models")
    args = ap.parse_args()
    if args.stocks == "all":
        ids = sorted(int(d.removeprefix("stock_id=")) for d in os.listdir("book_train.parquet"))
    else:
        ids = [int(x) for x in args.stocks.split(",")]
    main(ids, with_nn=not args.no_nn)
