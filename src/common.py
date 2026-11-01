"""ค่ากลางที่ทุกสคริปต์ใช้ร่วมกัน: ตำแหน่งไฟล์ การโหลดราคา และรายชื่อหุ้น"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

# yfinance ยังใช้ inplace= ที่ pandas 3.x เตือนว่าจะเลิกใช้ ปิดไว้ไม่ให้รก output
warnings.filterwarnings("ignore", message=".*inplace keyword.*")

ROOT = Path(__file__).resolve().parent.parent
PRICES_DIR = ROOT / "data" / "prices"
UNIVERSE_DIR = ROOT / "universe"
EXAMPLES_DIR = ROOT / "examples"
SETUPS_DIR = ROOT / "setups"
OUTPUT_DIR = ROOT / "output"

EXAMPLES_CSV = EXAMPLES_DIR / "examples.csv"
PRICE_META = PRICES_DIR / "_meta.json"

# มูลค่าซื้อขายเฉลี่ย 20 วันขั้นต่ำ (หน่วยสกุลเงินของตลาดนั้น) ต่ำกว่านี้เข้าออกยาก ไม่สแกน
MIN_TRADED_VALUE = {"SET": 5_000_000, "US": 5_000_000}


def to_symbol(ticker: str, market: str) -> str:
    """ชื่อหุ้นที่คนใช้ → ชื่อที่ Yahoo Finance ใช้"""
    ticker = ticker.strip().upper()
    if market == "SET":
        return ticker if ticker.endswith(".BK") else f"{ticker}.BK"
    return ticker.replace(".", "-")


WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))}


def price_path(symbol: str) -> Path:
    # Windows ห้ามตั้งชื่อไฟล์ขึ้นต้นด้วย COM7, CON ฯลฯ (เช่นหุ้น COM7) จึงเติม _ นำหน้า
    name = f"_{symbol}" if symbol.split(".")[0].upper() in WINDOWS_RESERVED else symbol
    return PRICES_DIR / f"{name}.csv"


def load_prices(symbol: str) -> pd.DataFrame | None:
    path = price_path(symbol)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df[df["Close"] > 0]


def load_universe(markets: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for market in markets or ["SET", "US"]:
        path = UNIVERSE_DIR / f"{market.lower()}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, dtype=str))
    if not frames:
        raise SystemExit("ยังไม่มีรายชื่อหุ้น รัน: python src/universe.py")
    return pd.concat(frames, ignore_index=True)


def load_all_prices(markets: list[str] | None = None) -> dict[str, tuple[pd.Series, pd.DataFrame]]:
    """คืน {symbol: (แถว universe, ราคา)} เฉพาะหุ้นที่ดึงราคามาแล้ว"""
    out = {}
    for _, row in load_universe(markets).iterrows():
        df = load_prices(row["symbol"])
        if df is not None and len(df) >= 260:
            out[row["symbol"]] = (row, df)
    return out


def price_meta() -> dict:
    if PRICE_META.exists():
        return json.loads(PRICE_META.read_text(encoding="utf-8"))
    return {}


def load_examples() -> pd.DataFrame:
    cols = ["setup", "ticker", "market", "base_start", "entry_date", "label", "image", "note"]
    if not EXAMPLES_CSV.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(EXAMPLES_CSV, dtype=str).fillna("")
    return df[cols]
