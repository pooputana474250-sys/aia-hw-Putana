"""โหลดราคาและคำนวณตัววัดของทุกหุ้นในครั้งเดียว ให้ scan / learn_rule / backtest ใช้ร่วมกัน"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common import MIN_TRADED_VALUE, load_all_prices
from detector import FEATURES, compute_features


@dataclass
class Stock:
    symbol: str
    ticker: str
    market: str
    name: str
    prices: pd.DataFrame
    features: pd.DataFrame


def load_market(markets: list[str] | None = None) -> dict[str, Stock]:
    out = {}
    for symbol, (row, df) in load_all_prices(markets).items():
        out[symbol] = Stock(symbol, row["ticker"], row["market"], row.get("name", "") or "",
                            df, compute_features(df))
    print(f"โหลด {len(out)} หุ้น")
    return out


def find_stock(stocks: dict[str, Stock], ticker: str, market: str) -> Stock | None:
    from common import to_symbol
    return stocks.get(to_symbol(ticker, market))


def is_liquid(stock: Stock, f: pd.DataFrame) -> pd.Series:
    return f["traded_value"] >= MIN_TRADED_VALUE[stock.market]


def baseline_sample(stocks: dict[str, Stock], n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """วันสุ่มของหุ้นทั่วไป (ที่สภาพคล่องพอ) ใช้เป็น "ตลาดปกติ" เพื่อเทียบกับตัวอย่าง"""
    rng = np.random.default_rng(seed)
    per_stock = max(1, n // max(1, len(stocks)))
    rows = []
    for s in stocks.values():
        f = s.features.dropna(subset=list(FEATURES))
        f = f[is_liquid(s, f)]
        if len(f):
            rows.append(f.iloc[rng.choice(len(f), size=min(per_stock, len(f)), replace=False)])
    return pd.concat(rows) if rows else pd.DataFrame(columns=list(FEATURES))
