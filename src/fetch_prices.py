"""ดึงราคารายวัน (ปรับ split/ปันผลแล้ว) จาก Yahoo Finance → data/prices/<symbol>.csv

ใช้:  python src/fetch_prices.py               ดึงทุกหุ้นใน universe
      python src/fetch_prices.py --market SET  ดึงเฉพาะตลาดเดียว
      python src/fetch_prices.py --tickers PTT.BK AAPL
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd
import yfinance as yf

from common import PRICE_META, PRICES_DIR, load_universe, price_meta, price_path

BATCH = 50


def save_batch(symbols: list[str], period: str) -> list[str]:
    data = yf.download(symbols, period=period, auto_adjust=True, group_by="ticker",
                       threads=True, progress=False)
    saved = []
    for s in symbols:
        if isinstance(data.columns, pd.MultiIndex):
            if s not in data.columns.get_level_values(0):
                continue
            df = data[s]
        else:
            df = data
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(df):
            df.index.name = "Date"
            df.round(4).to_csv(price_path(s))
            saved.append(s)
    return saved


def fetch(symbols: list[str], period: str = "5y") -> None:
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    meta = price_meta()
    stamp = datetime.now().isoformat(timespec="minutes")
    missing = []
    for i in range(0, len(symbols), BATCH):
        batch = symbols[i:i + BATCH]
        saved = save_batch(batch, period)
        missing += [s for s in batch if s not in saved]
        for s in saved:
            meta[s] = {"retrieved": stamp, "source": "Yahoo Finance via yfinance (auto_adjust=True)"}
        print(f"  {min(i + BATCH, len(symbols))}/{len(symbols)}")
    PRICE_META.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"ดึงสำเร็จ {len(symbols) - len(missing)} หุ้น")
    if missing:
        print("ดึงไม่ได้:", " ".join(missing))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["SET", "US"])
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--period", default="5y")
    a = ap.parse_args()
    symbols = a.tickers or load_universe([a.market] if a.market else None)["symbol"].tolist()
    fetch(symbols, a.period)


if __name__ == "__main__":
    main()
