"""ทดสอบย้อนหลังทุก set-up บนทั้งตลาด → output/backtest.json

กติกาจำลอง
- สัญญาณ = วันที่ผ่าน set-up และสภาพคล่องพอ หุ้นเดิมต้องเว้น 20 วันก่อนนับสัญญาณใหม่
- เข้า = ราคาเปิดวันถัดไป · stop = ขอบล่างฐาน 60 วัน × 0.99
- วัดผลที่ 20 และ 60 วัน ถ้าโดน stop ก่อน ใช้ผลขาดทุนที่ stop
- R = กำไร / ความเสี่ยงต่อหุ้น (entry − stop)
- เทียบกับ "ตลาดปกติ": ซื้อหุ้นเดียวกันทุก 20 วันแบบไม่เลือก ถือ 60 วัน
- แบ่งช่วง: 60% แรกของช่วงข้อมูล กับ 40% หลัง และสำหรับสูตรที่เรียนจากตัวอย่าง
  แยกสัญญาณที่อยู่ห่างจากวันของตัวอย่างเกิน 60 วัน (out-of-sample) ออกมาให้ดูต่างหาก

ใช้:  python src/backtest.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import OUTPUT_DIR, load_examples, to_symbol
from market import Stock, is_liquid, load_market
from setups import evaluate, load_setups

COOLDOWN = 20
HORIZONS = (20, 60)


def trade_outcomes(s: Stock, signal_idx: np.ndarray) -> list[dict]:
    o, l, c = (s.prices[k].to_numpy() for k in ("Open", "Low", "Close"))
    stops = s.features["base_low"].to_numpy() * 0.99
    out = []
    for i in signal_idx:
        e = i + 1
        if e + max(HORIZONS) >= len(c):
            continue
        entry, stop = o[e], stops[i]
        if not (stop < entry):
            continue
        risk = entry - stop
        rec = {"date": s.prices.index[i], "ticker": s.ticker}
        for h in HORIZONS:
            lows = l[e:e + h + 1]
            hit = np.nonzero(lows <= stop)[0]
            exit_px = stop if len(hit) else c[e + h]
            rec[f"ret{h}"] = exit_px / entry - 1
            rec[f"r{h}"] = (exit_px - entry) / risk
        rec["risk_pct"] = risk / entry
        out.append(rec)
    return out


def pick_signals(mask: np.ndarray) -> np.ndarray:
    idx, last = [], -10**9
    for i in np.nonzero(mask)[0]:
        if i - last >= COOLDOWN:
            idx.append(i)
            last = i
    return np.array(idx, dtype=int)


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0}
    out = {"n": int(len(trades))}
    for h in HORIZONS:
        r = trades[f"ret{h}"]
        wins, losses = r[r > 0], r[r <= 0]
        out[f"win{h}"] = float((r > 0).mean())
        out[f"avg{h}"] = float(r.mean())
        out[f"avg_win{h}"] = float(wins.mean()) if len(wins) else 0.0
        out[f"avg_loss{h}"] = float(losses.mean()) if len(losses) else 0.0
        out[f"avgR{h}"] = float(trades[f"r{h}"].mean())
    out["median_risk_pct"] = float(trades["risk_pct"].median())
    return out


def baseline(stocks: list[Stock]) -> dict:
    rows = []
    for s in stocks:
        c = s.prices["Close"].to_numpy()
        liquid = is_liquid(s, s.features).to_numpy()
        for i in range(260, len(c) - 61, COOLDOWN):
            if liquid[i]:
                rows.append(c[i + 61] / c[i + 1] - 1)
    r = pd.Series(rows)
    return {"n": int(len(r)), "win60": float((r > 0).mean()), "avg60": float(r.mean())} if len(r) else {"n": 0}


def run(stocks: dict[str, Stock]) -> dict:
    examples = load_examples()
    results = {}
    by_market = {m: [s for s in stocks.values() if s.market == m] for m in ("SET", "US")}
    base = {m: baseline(lst) for m, lst in by_market.items()}
    for setup in load_setups():
        if setup["type"] not in ("builtin", "rule"):
            continue   # ทรงวาด/ตัวอย่างที่ยังไม่มีสูตร ยังไม่มีเงื่อนไขให้จำลองสัญญาณย้อนหลัง
        ex = examples[examples["setup"] == setup["name"]]
        ex_dates = {}
        for _, e in ex.iterrows():
            ex_dates.setdefault(to_symbol(e["ticker"], e["market"]), []).append(pd.Timestamp(e["entry_date"]))
        res = {}
        for market, lst in by_market.items():
            trades = []
            for s in lst:
                passed, _ = evaluate(setup, s.features)
                mask = (passed & is_liquid(s, s.features)).fillna(False).to_numpy()
                for t in trade_outcomes(s, pick_signals(mask)):
                    near = [abs((t["date"] - d).days) for d in ex_dates.get(s.symbol, [])]
                    t["in_sample"] = bool(near) and min(near) <= 90
                    trades.append(t)
            df = pd.DataFrame(trades)
            entry = {"all": summarize(df), "baseline": base[market]}
            if not df.empty:
                split = df["date"].min() + (df["date"].max() - df["date"].min()) * 0.6
                entry["first_60pct"] = summarize(df[df["date"] < split])
                entry["last_40pct"] = summarize(df[df["date"] >= split])
                entry["split_date"] = str(split.date())
                entry["period"] = [str(df["date"].min().date()), str(df["date"].max().date())]
                if setup["type"] == "rule":
                    entry["out_of_sample"] = summarize(df[~df["in_sample"]])
            res[market] = entry
        results[setup["name"]] = res
        a = {m: res[m]["all"] for m in res}
        print(f"{setup['name']}: " + " · ".join(
            f"{m} n={v['n']}" + (f" win60={v['win60']:.0%} avg60={v['avg60']:+.1%} R60={v['avgR60']:+.2f}"
                                 if v["n"] else "") for m, v in a.items()))
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "backtest.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    for m, b in base.items():
        if b["n"]:
            print(f"ตลาดปกติ {m}: win60={b['win60']:.0%} avg60={b['avg60']:+.1%}")
    return results


if __name__ == "__main__":
    run(load_market())
