"""ชั้น C: สรุปรูปตัวอย่างของ set-up หนึ่งเป็นสูตรสแกนที่อ่านเป็นประโยคได้

วิธีคิด
1. วัดตัววัดชั้น A ของตัวอย่าง good ทุกรูป ณ วันสัญญาณ
2. เทียบกับวันสุ่มของหุ้นทั่วไปในตลาด (baseline) หาตัววัดที่ตัวอย่างมีร่วมกันแต่ตลาดมีน้อย
3. ตั้งเกณฑ์ให้ตัวอย่าง good ผ่านทุกรูป (< 10 รูป) หรือ 90% (≥ 10 รูป) แล้วเลือกเงื่อนไขทีละข้อ
   ข้อที่ตัด baseline ได้มากที่สุดก่อน จนเหลือหุ้นทั่วไปผ่าน ≤ 2% หรือครบ 6 ข้อ
4. ถ้าตัวอย่าง bad ยังผ่านสูตร หาเงื่อนไขเพิ่มที่ตัด bad ออกได้โดย good ยังผ่านครบ
5. ตั้งแต่ 10 รูปขึ้นไป ลอง decision tree ลึก 3 ชั้นด้วย ใช้ถ้าตัด baseline ได้ดีกว่าและ good หลุดไม่เกิน 20%

ใช้:  python src/learn_rule.py --setup my-base [--title "ชื่อที่อ่านง่าย"]
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

from common import OUTPUT_DIR, SETUPS_DIR, load_examples
from detector import FEATURES, feature_value_text
from market import Stock, baseline_sample, find_stock, load_market
from setups import condition_mask, condition_text
from similarity import signal_index

TARGET_BASELINE_RATE = 0.02
MAX_CONDITIONS = 6


def example_features(stocks: dict[str, Stock], ex: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, e in ex.iterrows():
        s = find_stock(stocks, e["ticker"], e["market"])
        if s is None:
            print(f"  ข้าม {e['ticker']}: ไม่มีราคา (เพิ่มใน universe แล้วรัน fetch_prices ก่อน)")
            continue
        sig = signal_index(s.prices, e["entry_date"])
        r = s.features.iloc[sig][list(FEATURES)].copy()
        r["ticker"], r["label"], r["signal_date"] = e["ticker"], e["label"], s.prices.index[sig].date()
        rows.append(r)
    return pd.DataFrame(rows)


def candidate_conditions(good: pd.DataFrame, base: pd.DataFrame) -> list[dict]:
    """เงื่อนไขทางเดียวต่อหนึ่งตัววัด ตั้งให้ good ผ่านเกือบหมด แล้ววัดว่าตลาดผ่านกี่ %"""
    strict = len(good) < 10
    out = []
    for feat in FEATURES:
        g, b = good[feat].dropna(), base[feat].dropna()
        if len(g) < max(2, int(0.8 * len(good))) or b.empty:
            continue
        # เผื่อขอบเล็กน้อย (5% ของ IQR ตลาด) กันสูตรแคบเกินเท่าตัวอย่างพอดี
        slack = 0.05 * (b.quantile(0.75) - b.quantile(0.25))
        if g.median() <= b.median():
            value = (g.max() if strict else g.quantile(0.9)) + slack
            op = "<="
        else:
            value = (g.min() if strict else g.quantile(0.1)) - slack
            op = ">="
        cond = {"feature": feat, "op": op, "value": float(value)}
        cond["baseline_pass"] = float(condition_mask(base, cond).mean())
        cond["good_pass"] = float(condition_mask(good, cond).mean())
        out.append(cond)
    return sorted(out, key=lambda c: c["baseline_pass"])


def pass_rate(frame: pd.DataFrame, conds: list[dict]) -> float:
    if not conds or frame.empty:
        return 1.0
    m = pd.Series(True, index=frame.index)
    for c in conds:
        m &= condition_mask(frame, c)
    return float(m.mean())


def greedy_rule(cands: list[dict], good: pd.DataFrame, bad: pd.DataFrame, base: pd.DataFrame) -> list[dict]:
    chosen: list[dict] = []
    rate = 1.0
    for c in cands:
        if len(chosen) >= MAX_CONDITIONS or rate <= TARGET_BASELINE_RATE:
            break
        if c["baseline_pass"] > 0.7:
            continue
        new_rate = pass_rate(base, chosen + [c])
        if new_rate < rate * 0.85:   # ต้องตัดตลาดได้อีกอย่างน้อย 15% ถึงจะคุ้มเพิ่มเงื่อนไข
            chosen.append(c)
            rate = new_rate
    # ตัวอย่าง bad ที่ยังผ่าน: หาเงื่อนไขที่ตัดมันออกโดย good ยังผ่านเท่าเดิม
    for i in range(len(bad)):
        one = bad.iloc[[i]]
        if pass_rate(one, chosen) == 0:
            continue
        good_rate = pass_rate(good, chosen)
        for c in cands:
            if c in chosen:
                continue
            if pass_rate(one, chosen + [c]) == 0 and pass_rate(good, chosen + [c]) >= good_rate:
                chosen.append(c)
                break
    return chosen


def tree_rule(good: pd.DataFrame, bad: pd.DataFrame, base: pd.DataFrame) -> list[dict] | None:
    """เส้นทางใน decision tree ที่ไปจบใบที่มี good มากที่สุด แปลงเป็นเงื่อนไข"""
    from sklearn.tree import DecisionTreeClassifier

    feats = list(FEATURES)
    neg = pd.concat([bad[feats], base[feats]])
    X = pd.concat([good[feats], neg]).fillna(neg.median())
    y = np.r_[np.ones(len(good)), np.zeros(len(neg))]
    w = np.r_[np.full(len(good), len(neg) / max(1, len(good))), np.ones(len(neg))]
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=0).fit(X, y, sample_weight=w)
    t = clf.tree_
    leaves = clf.apply(X.iloc[:len(good)])
    best_leaf = np.bincount(leaves).argmax()

    def path(node: int, target: int, acc: list[dict]) -> list[dict] | None:
        if node == target:
            return acc
        if t.children_left[node] == -1:
            return None
        f, thr = feats[t.feature[node]], float(t.threshold[node])
        return (path(t.children_left[node], target, acc + [{"feature": f, "op": "<=", "value": thr}])
                or path(t.children_right[node], target, acc + [{"feature": f, "op": ">=", "value": thr}]))

    conds = path(0, best_leaf, [])
    if conds:
        for c in conds:
            c["baseline_pass"] = float(condition_mask(base, c).mean())
            c["good_pass"] = float(condition_mask(good, c).mean())
    return conds


def write_report(name: str, title: str, conds: list[dict], ex: pd.DataFrame, base: pd.DataFrame,
                 method: str) -> None:
    good, bad = ex[ex["label"] == "good"], ex[ex["label"] == "bad"]
    lines = [
        "---", f"title: สูตรสแกนที่เรียนจากรูปตัวอย่าง · {title}", "type: note",
        f"created: {date.today()}", f"updated: {date.today()}", "tags: [screener, setup, learned-rule]",
        "sources: [Yahoo Finance via yfinance]", "status: draft", "---", "",
        f"# {title}", "",
        f"วิธีสร้าง: {method} · ตัวอย่าง good {len(good)} รูป · bad {len(bad)} รูป · "
        f"เทียบกับวันสุ่มของหุ้นทั่วไป {len(base):,} วัน", "",
        "## สูตร", "",
        "| เงื่อนไข | ตัวอย่าง good ผ่าน | หุ้นทั่วไปผ่าน |", "|---|---|---|",
    ]
    for c in conds:
        lines.append(f"| {condition_text(c)} | {c['good_pass']:.0%} | {c['baseline_pass']:.1%} |")
    lines += ["", f"ผ่านครบทุกข้อ: ตัวอย่าง good {pass_rate(good, conds):.0%} · "
                  f"bad {pass_rate(bad, conds) if len(bad) else float('nan'):.0%} · "
                  f"หุ้นทั่วไป {pass_rate(base, conds):.2%}", "",
              "## ค่าของแต่ละตัวอย่าง", "",
              "| หุ้น | ป้าย | วันสัญญาณ | " + " | ".join(FEATURES[c['feature']] for c in conds) + " |",
              "|---|---|---|" + "---|" * len(conds)]
    for _, r in ex.iterrows():
        vals = " | ".join(feature_value_text(c["feature"], r[c["feature"]]) for c in conds)
        lines.append(f"| {r['ticker']} | {r['label']} | {r['signal_date']} | {vals} |")
    if len(good) < 5:
        lines += ["", "> ⚠️ ตัวอย่างน้อยกว่า 5 รูป สูตรอาจจำแค่ตัวอย่าง (overfit) ให้ดูผลทดสอบย้อนหลังและแกลเลอรีประกอบเสมอ"]
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / f"rule_report_{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def learn(name: str, title: str | None, stocks: dict[str, Stock] | None = None) -> dict:
    ex = load_examples()
    ex = ex[ex["setup"] == name]
    if ex.empty:
        raise SystemExit(f"ยังไม่มีตัวอย่างของ set-up '{name}' ใน examples/examples.csv")
    stocks = stocks or load_market()
    feats = example_features(stocks, ex)
    good, bad = feats[feats["label"] == "good"], feats[feats["label"] == "bad"]
    if len(good) < 3:
        raise SystemExit(f"มีตัวอย่าง good {len(good)} รูป ต้องมีอย่างน้อย 3 รูปถึงจะสรุปสูตรได้ "
                         "(ระหว่างนี้ scan.py ใช้ชั้น B หาหุ้นที่คล้ายรูปตัวอย่างให้แล้ว)")
    base = baseline_sample(stocks)
    cands = candidate_conditions(good, base)
    conds, method = greedy_rule(cands, good, bad, base), "เลือกเงื่อนไขทีละข้อจากช่วงค่าของตัวอย่าง"
    if len(good) >= 10:
        tconds = tree_rule(good, bad, base)
        if tconds and pass_rate(good, tconds) >= 0.8 and pass_rate(base, tconds) < pass_rate(base, conds):
            conds, method = tconds, "decision tree ลึก 3 ชั้น"
    title = title or name
    setup = {
        "name": name, "title": title, "type": "rule", "method": method,
        "learned": str(date.today()),
        "conditions": [{k: c[k] for k in ("feature", "op", "value")} | {"text": condition_text(c)} for c in conds],
        "stats": {"good": len(good), "bad": len(bad), "good_pass": pass_rate(good, conds),
                  "bad_pass": pass_rate(bad, conds) if len(bad) else None,
                  "baseline_pass": pass_rate(base, conds), "baseline_n": len(base)},
        "description": f"เรียนจากรูปตัวอย่าง good {len(good)} รูป bad {len(bad)} รูป ด้วยวิธี{method}",
    }
    SETUPS_DIR.mkdir(exist_ok=True)
    (SETUPS_DIR / f"{name}.json").write_text(json.dumps(setup, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(name, title, conds, feats, base, method)
    print(f"สูตร '{title}' ({method}):")
    for c in conds:
        print(f"  - {condition_text(c)}")
    print(f"good ผ่าน {setup['stats']['good_pass']:.0%} · หุ้นทั่วไปผ่าน {setup['stats']['baseline_pass']:.2%}")
    print(f"→ setups/{name}.json · output/rule_report_{name}.md")
    return setup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True)
    ap.add_argument("--title")
    a = ap.parse_args()
    learn(a.setup, a.title)


if __name__ == "__main__":
    main()
