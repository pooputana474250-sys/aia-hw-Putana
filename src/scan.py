"""สแกนทุก set-up → output/signals.csv และ output/dashboard.html

ใช้:  python src/scan.py                    สแกนวันล่าสุดที่มีข้อมูล
      python src/scan.py --date 2025-06-02  สแกนย้อนไปวันนั้น (ใช้ตรวจว่าระบบเจอตัวอย่างเองไหม)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import pandas as pd

from common import MIN_TRADED_VALUE, OUTPUT_DIR, ROOT, load_examples, price_meta
from detector import FEATURES, builtin_score, feature_value_text
from market import Stock, find_stock, load_market
from setups import evaluate, load_setups, reasons
from similarity import Template, build_template, feature_scale, similarity_scores, sketch_scores

TOP_PER_MARKET = 40
TOP_SIMILAR = 15
CHART_BARS = 180


def bars(df: pd.DataFrame, max_bars: int = 300) -> dict:
    """ราคาสำหรับวาดกราฟ ถ้ายาวเกิน max_bars แท่ง รวมเป็นแท่งรายสัปดาห์ (วันที่ = วันทำการสุดท้ายของสัปดาห์)"""
    if len(df) > max_bars:
        g = df.assign(Date=df.index).groupby(df.index.to_period("W"))
        df = pd.DataFrame({"Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
                           "Close": g["Close"].last(), "Volume": g["Volume"].sum(), "Date": g["Date"].last()})
        df = df.set_index("Date")
    return {"d": [d.strftime("%Y-%m-%d") for d in df.index],
            **{k[0].lower(): [round(float(x), 4) for x in df[k]] for k in ("Open", "High", "Low", "Close")},
            "v": [float(x) for x in df["Volume"]]}


def templates_for(name: str, stocks: dict[str, Stock], before: pd.Timestamp) -> list[Template]:
    out = []
    for _, e in load_examples().query("setup == @name").iterrows():
        s = find_stock(stocks, e["ticker"], e["market"])
        if s is None:
            print(f"  ข้ามตัวอย่าง {e['ticker']}: ยังไม่มีราคา")
            continue
        t = build_template(e["ticker"], e["label"], s.prices, s.features, e["base_start"], e["entry_date"])
        # ตอนสแกนย้อนหลัง ใช้ได้เฉพาะตัวอย่างที่เกิดก่อนวันสแกน ไม่งั้นเท่ากับรู้อนาคต
        if t.signal_date <= before:
            out.append(t)
    return out


def scan_setup(setup: dict, stocks: dict[str, Stock], end_idx: dict[str, int], today: pd.DataFrame,
               scale: pd.Series, scan_date: pd.Timestamp) -> tuple[list[dict], list[dict]]:
    templates = templates_for(setup["name"], stocks, scan_date)
    goods = [t for t in templates if t.label == "good"]
    bads = [t for t in templates if t.label == "bad"]
    liquid = today[today["liquid"]]

    passed, score = evaluate(setup, liquid)
    _, all_parts = builtin_score(liquid)   # คำนวณครั้งเดียว แทนการคำนวณทีละหุ้นในลูปด้านล่าง
    table = pd.DataFrame({"passed": passed, "score": score})
    if goods:
        prices = {s: stocks[s].prices for s in liquid.index}
        sims = [similarity_scores(t, prices, end_idx, liquid, scale) for t in goods]
        stacked = pd.concat([x["similarity"] for x in sims], axis=1)
        table["similarity"] = stacked.max(axis=1)
        best = stacked.to_numpy().argmax(axis=1)
        table["best_example"] = [goods[i].ticker for i in best]
        table["window"] = [sims[i].loc[s, "window"] for s, i in zip(stacked.index, best)]
        for part in ("sim_shape", "sim_volume", "sim_features"):
            table[part] = [sims[i].loc[s, part] for s, i in zip(stacked.index, best)]
        if bads:
            bad_sim = pd.concat([similarity_scores(t, prices, end_idx, liquid, scale)["similarity"]
                                 for t in bads], axis=1).max(axis=1)
            table["like_bad"] = bad_sim > table["similarity"]
            table.loc[table["like_bad"], "similarity"] *= 0.7
    if setup.get("sketch") and not goods:
        prices = {s: stocks[s].prices for s in liquid.index}
        sk = sketch_scores(setup["sketch"], setup["length"], prices, end_idx)
        table["similarity"] = sk["similarity"]
        table["sim_shape"] = sk["similarity"]
        table["sim_volume"] = float("nan")
        table["sim_features"] = float("nan")
        table["best_example"] = "ทรงที่วาด"
        table["window"] = sk["window"]
    has_sim = "similarity" in table
    rank_col = "similarity" if has_sim else "score"

    signals = []
    for market in ("SET", "US"):
        sub = table[[stocks[s].market == market for s in table.index]]
        chosen = sub[sub["passed"]].sort_values(rank_col, ascending=False).head(TOP_PER_MARKET)
        if has_sim:
            similar = sub[~sub["passed"]].sort_values("similarity", ascending=False).head(TOP_SIMILAR)
            chosen = pd.concat([chosen, similar])
        for sym, r in chosen.iterrows():
            s, f = stocks[sym], liquid.loc[sym]
            i = end_idx[sym]
            base_low, close = float(f["base_low"]), float(s.prices["Close"].iloc[i])
            base_days = int(f["base_days"]) if pd.notna(f["base_days"]) else 0
            signals.append({
                "setup": setup["name"], "ticker": s.ticker, "market": market, "name": s.name,
                "date": s.prices.index[i].strftime("%Y-%m-%d"), "close": close,
                "passed": bool(r["passed"]), "score": round(float(r["score"]), 1),
                "similarity": round(float(r["similarity"]), 1) if has_sim else None,
                "sim_parts": {k: (None if pd.isna(r[k]) else round(float(r[k]), 1))
                              for k in ("sim_shape", "sim_volume", "sim_features")} if has_sim else None,
                "best_example": r.get("best_example") if has_sim else None,
                "like_bad": bool(r.get("like_bad", False)),
                "stop": round(base_low * 0.99, 4), "risk_pct": round(1 - base_low * 0.99 / close, 4),
                "base_low": base_low, "base_top": float(f["base_top"]),
                "base_start": s.prices.index[max(0, i - base_days)].strftime("%Y-%m-%d"),
                "features": {k: feature_value_text(k, f[k]) for k in FEATURES},
                "reasons": reasons(setup, f, all_parts.loc[sym]),
                "tv": s.tv,
                "window": int(r["window"]) if has_sim and pd.notna(r.get("window")) else None,
                # ทรงยาวให้เห็นทั้งช่วงที่จับคู่ได้ + ก่อนหน้าอีกนิด
                "chart": bars(s.prices.iloc[max(0, i + 1 - max(CHART_BARS, int(r.get("window", 0) or 0) + 40)):i + 1]),
            })
    examples = [{"ticker": t.ticker, "label": t.label, "signal_date": t.signal_date.strftime("%Y-%m-%d"),
                 "chart": bars(t.chart)} for t in templates]
    return signals, examples


def latest_date(stocks: dict[str, Stock]) -> pd.Timestamp:
    return max(s.prices.index[-1] for s in stocks.values())


def prepare(stocks: dict[str, Stock], scan_date: pd.Timestamp):
    """ตำแหน่งแถวของวันสแกนในแต่ละหุ้น + ตัววัดของวันนั้น + สเกลของตัววัด"""
    end_idx, rows = {}, []
    for sym, s in stocks.items():
        i = int(s.prices.index.searchsorted(scan_date, side="right")) - 1
        # ข้ามหุ้นที่ไม่มีราคาในช่วง 7 วันก่อนวันสแกน (หยุดซื้อขาย/ถูกถอด)
        if i < 260 or (scan_date - s.prices.index[i]).days > 7:
            continue
        end_idx[sym] = i
        row = s.features.iloc[i].copy()
        row["liquid"] = row["traded_value"] >= MIN_TRADED_VALUE[s.market]
        rows.append(row.rename(sym))
    today = pd.DataFrame(rows)
    today["liquid"] = today["liquid"].astype(bool)
    return end_idx, today, feature_scale(today[today["liquid"]])


def render(payload: dict) -> None:
    template = (ROOT / "src" / "dashboard_template.html").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    (OUTPUT_DIR / "dashboard.html").write_text(template.replace("__DATA__", data), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    a = ap.parse_args()
    stocks = load_market()
    scan_date = pd.Timestamp(a.date) if a.date else latest_date(stocks)
    end_idx, today, scale = prepare(stocks, scan_date)

    backtest_path = OUTPUT_DIR / "backtest.json"
    backtest = json.loads(backtest_path.read_text(encoding="utf-8")) if backtest_path.exists() else {}
    all_signals, setups_out = [], []
    for setup in load_setups():
        signals, examples = scan_setup(setup, stocks, end_idx, today, scale, scan_date)
        all_signals += signals
        setups_out.append({**setup, "examples": examples, "backtest": backtest.get(setup["name"])})
        n_pass = sum(x["passed"] for x in signals)
        print(f"{setup['name']}: ผ่านสูตร {n_pass} ตัว · แสดงทั้งหมด {len(signals)} ตัว")

    OUTPUT_DIR.mkdir(exist_ok=True)
    cols = ["setup", "market", "ticker", "date", "passed", "score", "similarity", "best_example",
            "close", "stop", "risk_pct", "base_start"]
    pd.DataFrame(all_signals, columns=cols + ["chart"])[cols].to_csv(OUTPUT_DIR / "signals.csv", index=False,
                                                                     encoding="utf-8-sig")
    meta = price_meta()
    retrieved = sorted({v["retrieved"] for v in meta.values()})
    render({
        "scan_date": scan_date.strftime("%Y-%m-%d"),
        "generated": datetime.now().isoformat(timespec="minutes"),
        "source": "Yahoo Finance ผ่าน yfinance (ราคาปรับ split/ปันผลแล้ว)",
        "retrieved": retrieved[-1] if retrieved else "",
        "universe": {"SET": sum(s.market == "SET" for s in stocks.values()),
                     "US": sum(s.market == "US" for s in stocks.values())},
        "setups": setups_out, "signals": all_signals,
    })
    print(f"→ output/signals.csv · output/dashboard.html (วันสแกน {scan_date.date()})")


if __name__ == "__main__":
    main()
