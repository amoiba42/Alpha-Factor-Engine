"""Data acquisition, validation and cleaning.

Conventions
-----------
* Long format: one row per (date, ticker) with columns
  open, high, low, close, adj_close, volume.
* Wide format: DataFrame indexed by date, one column per ticker.
* `close`/`open`/`high`/`low`/`volume` are split-adjusted (Yahoo convention);
  `adj_close` is additionally dividend-adjusted and is the only price used for returns.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

PRICE_COLS = ["open", "high", "low", "close", "adj_close", "volume"]


# --------------------------------------------------------------------------------------
# S&P 500 point-in-time membership
# --------------------------------------------------------------------------------------
def download_membership(url: str, dest: Path, timeout: int = 120) -> Path:
    """Download the historical S&P 500 components file (pinned commit) if not cached."""
    if dest.exists():
        return dest
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    pd.read_csv(StringIO(resp.text))  # fail loudly on a truncated download
    dest.write_text(resp.text)
    return dest


def read_membership_snapshots(path: Path) -> pd.Series:
    """Return a Series indexed by snapshot date whose values are lists of member tickers."""
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    return pd.Series(df["tickers"].str.split(",").to_numpy(), index=df["date"].to_numpy())


def membership_tickers(snapshots: pd.Series, start: str, end: str) -> list[str]:
    """All tickers that were members at any snapshot in [start, end], plus the snapshot
    in force at `start` (the last one on or before it)."""
    idx = snapshots.index
    first = idx[idx <= pd.Timestamp(start)]
    lo = first[-1] if len(first) else idx[0]
    sel = snapshots[(idx >= lo) & (idx <= pd.Timestamp(end))]
    return sorted({t.strip() for row in sel for t in row if t.strip()})


def membership_mask(snapshots: pd.Series, dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """Boolean (date x ticker) mask: True if the ticker was an index member on that date.

    Uses the most recent snapshot on or before each date (no future information).
    """
    long = snapshots.explode().rename("ticker").rename_axis("date").reset_index()
    long["ticker"] = long["ticker"].str.strip()
    long["member"] = True
    snap = long.pivot_table(index="date", columns="ticker", values="member", aggfunc="any")
    snap = snap.reindex(columns=tickers).fillna(False).astype(bool)
    union = snap.index.union(dates)
    mask = snap.reindex(union).ffill().reindex(dates)
    return mask.fillna(False).astype(bool)


# --------------------------------------------------------------------------------------
# Price download / loading
# --------------------------------------------------------------------------------------
def to_yahoo_symbol(ticker: str) -> str:
    return ticker.replace(".", "-")


def _download_batch(symbols: list[str], start: str, end_exclusive: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        symbols, start=start, end=end_exclusive, auto_adjust=False, actions=False,
        progress=False, threads=True, group_by="column",
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "ticker", *PRICE_COLS])
    raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                              "Adj Close": "adj_close", "Volume": "volume"}, level=0)
    long = raw.stack(level=1, future_stack=True).rename_axis(["date", "ticker"]).reset_index()
    long = long.dropna(subset=["adj_close", "close"], how="all")
    return long[["date", "ticker", *PRICE_COLS]]


def download_prices_yfinance(tickers: list[str], start: str, end: str, dest: Path,
                             batch_size: int = 50, max_retries: int = 2) -> pd.DataFrame:
    """Download daily OHLCV + Adj Close from Yahoo Finance and cache to parquet.

    Yahoo only serves currently-listed symbols, so most delisted historical members are
    missing; the manifest written next to the parquet records which tickers failed.
    """
    import yfinance as yf

    cache_dir = dest.parent / "yf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))

    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    sym_to_ticker = {to_yahoo_symbol(t): t for t in tickers}
    pending = list(sym_to_ticker)
    frames: list[pd.DataFrame] = []
    for attempt in range(max_retries + 1):
        failed: list[str] = []
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            try:
                part = _download_batch(batch, start, end_exclusive)
            except Exception as exc:  # network / rate-limit errors: retry whole batch later
                log.warning("batch %d failed: %s", i // batch_size, exc)
                failed.extend(batch)
                time.sleep(5)
                continue
            got = set(part["ticker"].unique())
            failed.extend(s for s in batch if s not in got)
            frames.append(part)
            log.info("attempt %d batch %d/%d: %d/%d symbols", attempt, i // batch_size + 1,
                     -(-len(pending) // batch_size), len(got), len(batch))
        pending = failed
        if not pending:
            break
        batch_size = max(5, batch_size // 5)  # smaller batches for retries
        time.sleep(10)

    prices = pd.concat(frames, ignore_index=True)
    prices["ticker"] = prices["ticker"].map(lambda s: sym_to_ticker.get(s, s))
    prices["date"] = pd.to_datetime(prices["date"]).dt.tz_localize(None).astype("datetime64[ns]")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices.to_parquet(dest, index=False)

    manifest = {
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "yfinance_version": yf.__version__,
        "start": start, "end": end,
        "requested": len(tickers),
        "received": int(prices["ticker"].nunique()),
        "failed": sorted(sym_to_ticker.get(s, s) for s in pending),
    }
    dest.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return prices


def load_raw_data(cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Return (raw long price panel incl. benchmark, membership snapshots).

    Downloads on first use and caches to `data/raw`; later runs read the cache so the
    results are reproducible from the same snapshot.
    """
    raw_dir: Path = cfg["paths"]["raw_dir"]
    dcfg = cfg["data"]
    snapshots = read_membership_snapshots(
        download_membership(dcfg["membership_url"], raw_dir / "sp500_membership.csv"))

    dest = raw_dir / "prices_yfinance.parquet"
    if dest.exists():
        prices = pd.read_parquet(dest)
    else:
        tickers = membership_tickers(snapshots, dcfg["start"], dcfg["end"])
        tickers = sorted(set(tickers) | {dcfg["benchmark"]})
        prices = download_prices_yfinance(tickers, dcfg["start"], dcfg["end"], dest,
                                          batch_size=dcfg["download_batch_size"])
    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")
    return prices, snapshots


# --------------------------------------------------------------------------------------
# Validation and cleaning
# --------------------------------------------------------------------------------------
def validate_prices(prices: pd.DataFrame) -> dict[str, int | float]:
    """Data-quality counts on the raw long panel (before cleaning). Nothing is modified."""
    p = prices.sort_values(["ticker", "date"])
    ohlc = p[["open", "high", "low", "close"]]
    ret = p.groupby("ticker")["adj_close"].pct_change(fill_method=None)
    ratio = p["adj_close"] / p["close"]
    ratio_chg = ratio.groupby(p["ticker"]).pct_change(fill_method=None).abs()
    obs = p.groupby("ticker")["adj_close"].count()
    return {
        "rows": len(p),
        "tickers": p["ticker"].nunique(),
        "dates": p["date"].nunique(),
        "first_date": str(p["date"].min().date()),
        "last_date": str(p["date"].max().date()),
        "duplicate_rows": int(p.duplicated(["date", "ticker"]).sum()),
        "missing_adj_close": int(p["adj_close"].isna().sum()),
        "missing_ohlc": int(ohlc.isna().any(axis=1).sum()),
        "missing_volume": int(p["volume"].isna().sum()),
        "zero_volume": int((p["volume"] == 0).sum()),
        "non_positive_prices": int((p[["open", "high", "low", "close", "adj_close"]] <= 0).any(axis=1).sum()),
        "high_below_low": int((p["high"] < p["low"]).sum()),
        "close_outside_range": int(((p["close"] > p["high"] * 1.0001) | (p["close"] < p["low"] * 0.9999)).sum()),
        "abs_return_gt_25pct": int((ret.abs() > 0.25).sum()),
        "abs_return_gt_50pct": int((ret.abs() > 0.50).sum()),
        "dividend_or_split_adjustment_events": int((ratio_chg > 1e-4).sum()),
        "tickers_lt_252_obs": int((obs < 252).sum()),
        "median_obs_per_ticker": float(obs.median()),
    }


def clean_prices(prices: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply deterministic cleaning rules. Returns (clean long panel, counts of actions).

    Rules (no forward-filling of prices, so missing days give missing returns):
    1. Drop exact duplicate (date, ticker) rows, keeping the last.
    2. Non-positive prices -> NaN. (No minimum-price rule: prices are split-adjusted.)
    3. Negative volume -> NaN.
    4. A daily adj_close return with |r| > `cleaning.max_abs_daily_return` is treated as a
       bad print: that day's prices are set to NaN (the move is not tradable as recorded).
    """
    c = cfg["cleaning"]
    p = prices.sort_values(["ticker", "date"]).copy()
    actions: dict[str, int] = {}

    n0 = len(p)
    p = p.drop_duplicates(["date", "ticker"], keep="last")
    actions["duplicates_dropped"] = n0 - len(p)

    price_cols = ["open", "high", "low", "close", "adj_close"]
    bad = p[price_cols] <= 0
    actions["non_positive_prices_set_nan"] = int(bad.to_numpy().sum())
    p[price_cols] = p[price_cols].mask(bad)

    neg_vol = p["volume"] < 0
    actions["negative_volume_set_nan"] = int(neg_vol.sum())
    p.loc[neg_vol, "volume"] = np.nan

    ret = p.groupby("ticker")["adj_close"].pct_change(fill_method=None)
    spike = ret.abs() > c["max_abs_daily_return"]
    actions["extreme_return_days_set_nan"] = int(spike.sum())
    p.loc[spike, price_cols] = np.nan

    p = p.dropna(subset=["adj_close"])
    actions["rows_after_cleaning"] = len(p)
    return p.reset_index(drop=True), actions


def to_wide(prices: pd.DataFrame, cols: list[str] = PRICE_COLS) -> dict[str, pd.DataFrame]:
    """Pivot the long panel into {column: date x ticker} frames on a common calendar."""
    wide = {}
    for col in cols:
        w = prices.pivot(index="date", columns="ticker", values=col).sort_index()
        wide[col] = w.astype(float)
    dates, tickers = wide["adj_close"].index, wide["adj_close"].columns
    return {k: v.reindex(index=dates, columns=tickers) for k, v in wide.items()}


def build_panel(cfg: dict) -> dict:
    """Load, validate and clean data. Returns a dict with wide price frames, the benchmark
    return series, the membership mask and data-quality reports."""
    raw, snapshots = load_raw_data(cfg)
    bench_sym = cfg["data"]["benchmark"]
    start, end = pd.Timestamp(cfg["data"]["start"]), pd.Timestamp(cfg["data"]["end"])
    raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]

    bench_raw = raw[raw["ticker"] == bench_sym]
    stocks_raw = raw[raw["ticker"] != bench_sym]
    quality_raw = validate_prices(stocks_raw)
    stocks, actions = clean_prices(stocks_raw, cfg)
    quality_clean = validate_prices(stocks)

    wide = to_wide(stocks)
    dates = wide["adj_close"].index
    bench = (bench_raw.set_index("date")["adj_close"].sort_index().reindex(dates)
             .pct_change(fill_method=None).rename(bench_sym))
    mask = membership_mask(snapshots, dates, list(wide["adj_close"].columns))
    return {
        "wide": wide,
        "benchmark_returns": bench,
        "membership": mask,
        "quality_raw": quality_raw,
        "quality_clean": quality_clean,
        "cleaning_actions": actions,
    }
