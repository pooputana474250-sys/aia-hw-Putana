"""ตรวจว่าระบบไม่โกงตัวเอง (รันหลังแก้ detector.py หรือ similarity.py ทุกครั้ง)

1. ไม่มีข้อมูลอนาคตรั่ว: ตัดราคาหลังวันหนึ่งทิ้ง แล้วตัววัดของวันนั้นต้องเท่าเดิม
2. หาตัวเองเจอ: ใช้หุ้นตัวหนึ่ง ณ วันหนึ่งเป็นรูปตัวอย่าง แล้วสแกนย้อนไปวันนั้น หุ้นตัวนั้นต้องติดอันดับ 1

ใช้:  python src/selftest.py
"""
from __future__ import annotations

import numpy as np

from detector import FEATURES, compute_features
from market import load_market
from similarity import build_template, feature_scale, similarity_scores

CASES = [("INTC", "US", "2025-04-08", "2025-08-05"), ("KCE.BK", "SET", "2023-05-26", "2023-07-11")]


def main() -> None:
    stocks = load_market()
    ok = True
    for symbol, market, base_start, entry in CASES:
        s = stocks[symbol]
        sig = int(s.prices.index.searchsorted(entry)) - 1
        full = s.features.iloc[sig][list(FEATURES)]
        cut = compute_features(s.prices.iloc[:sig + 1]).iloc[-1][list(FEATURES)]
        leak = not np.allclose(full.to_numpy(float), cut.to_numpy(float), equal_nan=True)
        print(f"[{'FAIL' if leak else 'ok'}] ไม่รู้อนาคต: {symbol} {s.prices.index[sig].date()}")
        ok &= not leak

        t = build_template(symbol, "good", s.prices, s.features, base_start, entry)
        end_idx = {k: int(v.prices.index.searchsorted(t.signal_date, side="right")) - 1 for k, v in stocks.items()}
        end_idx = {k: i for k, i in end_idx.items() if i >= 260}
        today = __import__("pandas").DataFrame({k: stocks[k].features.iloc[i] for k, i in end_idx.items()}).T
        sims = similarity_scores(t, {k: stocks[k].prices for k in end_idx}, end_idx, today, feature_scale(today))
        top = sims["similarity"].idxmax()
        print(f"[{'ok' if top == symbol else 'FAIL'}] หาตัวเองเจอ: อันดับ 1 = {top} ({sims['similarity'].max():.0f})")
        ok &= top == symbol
    print("ผ่านทุกข้อ" if ok else "มีข้อไม่ผ่าน")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
