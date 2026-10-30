"""Adjusted daily bars, exact-request caching, and common-calendar alignment."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def clean_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Sort and drop invalid bars without imputing any prices."""
    bars = bars.copy()
    bars.columns = [str(c).lower() for c in bars.columns]
    bars.index = pd.to_datetime(bars.index).tz_localize(None).normalize()
    bars = bars.sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]
    bars = bars[["open", "high", "low", "close", "volume"]].replace([np.inf, -np.inf], np.nan)
    valid = bars.notna().all(axis=1) & (bars[["open", "high", "low", "close"]] > 0).all(axis=1)
    return bars.loc[valid & (bars.volume >= 0)]


def download_bars(asset: str, start: str, end: str, cache_dir: str) -> pd.DataFrame:
    """Download auto-adjusted OHLCV; cache keys include dates and adjustment policy."""
    import yfinance as yf

    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(f"{asset}|{start}|{end}|auto_adjust=True|v1".encode()).hexdigest()[:12]
    path = directory / f"{asset}_{key}.parquet"
    if path.exists():
        return clean_bars(pd.read_parquet(path))
    bars = yf.download(asset, start=start, end=end, auto_adjust=True, progress=False,
                       multi_level_index=False)
    if bars.empty:
        raise ValueError(f"No data for {asset}; no synthetic fallback is used.")
    bars = clean_bars(bars)
    bars.to_parquet(path)
    return bars


def align_bars(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Intersect observed sessions. Never forward/backfill across market closures."""
    common = pd.DatetimeIndex(sorted(set.intersection(*(set(x.index) for x in data.values()))))
    if common.empty:
        raise ValueError("Assets have no common sessions")
    return {asset: bars.loc[common].copy() for asset, bars in data.items()}


def synthetic_bars(seed: int, periods: int = 1150) -> pd.DataFrame:
    """Offline smoke fixture only; never represented as market evidence."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=periods)
    opens = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.012, periods)))
    closes = opens * np.exp(rng.normal(0, 0.004, periods))
    return pd.DataFrame({"open": opens, "close": closes,
                         "high": np.maximum(opens, closes) * 1.003,
                         "low": np.minimum(opens, closes) * 0.997,
                         "volume": 1_000_000}, index=dates)
