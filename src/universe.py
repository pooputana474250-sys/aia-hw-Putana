"""สร้างรายชื่อหุ้นที่จะสแกน → universe/set.csv และ universe/us.csv

SET: ดึง SET100 จาก Wikipedia (เว็บ set.or.th ไม่ให้ดึงอัตโนมัติ)
     เพิ่มหุ้นเองได้ใน universe/set_extra.csv (คอลัมน์ ticker เดียว)
US:  S&P 500 + Nasdaq-100 จาก Wikipedia
"""
from __future__ import annotations

import io
from datetime import date

import pandas as pd
import requests

from common import UNIVERSE_DIR, to_symbol

HEADERS = {"User-Agent": "Mozilla/5.0 (aia-hw setup screener)"}
SOURCES = {
    "SET100": "https://en.wikipedia.org/wiki/SET100_Index",
    "SP500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "NDX": "https://en.wikipedia.org/wiki/Nasdaq-100",
    "NASDAQ_LISTED": "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "OTHER_LISTED": "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
}
# รหัสตลาดในไฟล์ otherlisted → ชื่อตลาดที่ TradingView ใช้นำหน้าชื่อหุ้น
TV_EXCHANGE = {"N": "NYSE", "A": "AMEX", "P": "AMEX", "Z": "BATS", "V": "IEX"}


def read_tables(url: str) -> list[pd.DataFrame]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_html(io.StringIO(r.text))


def first_table_with(tables: list[pd.DataFrame], column: str) -> pd.DataFrame:
    for t in tables:
        if column in t.columns:
            return t
    raise ValueError(f"ไม่เจอตารางที่มีคอลัมน์ {column}")


def build_set() -> pd.DataFrame:
    t = first_table_with(read_tables(SOURCES["SET100"]), "Symbol")
    rows = [{"ticker": s, "name": n, "source": SOURCES["SET100"]} for s, n in zip(t["Symbol"], t["Securities Name"])]
    extra = UNIVERSE_DIR / "set_extra.csv"
    if extra.exists():
        for s in pd.read_csv(extra, dtype=str)["ticker"].dropna():
            rows.append({"ticker": s.strip().upper(), "name": "", "source": "set_extra.csv"})
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df["market"] = "SET"
    return df


def build_us() -> pd.DataFrame:
    sp = first_table_with(read_tables(SOURCES["SP500"]), "Symbol")
    rows = [{"ticker": s, "name": n, "source": SOURCES["SP500"]} for s, n in zip(sp["Symbol"], sp["Security"])]
    # หน้า Nasdaq-100 เปลี่ยนโครงบ่อย ถ้าหาตารางรายชื่อไม่เจอ ใช้แค่ S&P 500 (ตัวใหญ่ส่วนมากซ้อนกันอยู่แล้ว)
    try:
        ndx = first_table_with(read_tables(SOURCES["NDX"]), "Ticker")
        rows += [{"ticker": s, "name": n, "source": SOURCES["NDX"]} for s, n in zip(ndx["Ticker"], ndx["Company"])]
    except ValueError:
        print("  ข้าม Nasdaq-100: ไม่เจอตารางรายชื่อในหน้า Wikipedia")
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df["market"] = "US"
    return df


def us_exchanges() -> dict[str, str]:
    """ชื่อหุ้นสหรัฐ → ตลาดที่จดทะเบียน (ไว้ทำชื่อแบบ NASDAQ:AAPL ให้ TradingView)"""
    out = {}
    for line in requests.get(SOURCES["NASDAQ_LISTED"], headers=HEADERS, timeout=30).text.splitlines()[1:]:
        out[line.split("|")[0]] = "NASDAQ"
    for line in requests.get(SOURCES["OTHER_LISTED"], headers=HEADERS, timeout=30).text.splitlines()[1:]:
        parts = line.split("|")
        if len(parts) > 2 and parts[2] in TV_EXCHANGE:
            out[parts[0]] = TV_EXCHANGE[parts[2]]
    return out


def tv_symbols(df: pd.DataFrame, market: str) -> list[str]:
    if market == "SET":
        return [f"SET:{t}" for t in df["ticker"]]
    try:
        ex = us_exchanges()
    except requests.RequestException:
        print("  ดึงตลาดของหุ้นสหรัฐไม่ได้ ใช้ชื่อหุ้นอย่างเดียวใน TradingView")
        ex = {}
    return [f"{ex[t]}:{t}" if t in ex else t for t in df["ticker"]]


def main() -> None:
    UNIVERSE_DIR.mkdir(exist_ok=True)
    today = date.today().isoformat()
    for market, builder in [("SET", build_set), ("US", build_us)]:
        df = builder()
        df["symbol"] = [to_symbol(t, market) for t in df["ticker"]]
        df["tv_symbol"] = tv_symbols(df, market)
        df["retrieved"] = today
        df = df[["ticker", "symbol", "tv_symbol", "market", "name", "source", "retrieved"]]
        df.to_csv(UNIVERSE_DIR / f"{market.lower()}.csv", index=False, encoding="utf-8")
        print(f"{market}: {len(df)} หุ้น → universe/{market.lower()}.csv")


if __name__ == "__main__":
    main()
