"""ชั้น A: แปลงกราฟเป็นตัววัด และให้คะแนนทรง "สะสมโซนล่างหลังขาลง"

ตัววัดทุกตัวของวัน t ใช้ข้อมูลถึงวัน t เท่านั้น (rolling ย้อนหลัง) จึงใช้ทดสอบย้อนหลังได้โดยไม่รู้อนาคต
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# ชื่อตัววัด → คำอธิบายภาษาไทย (ใช้ในสูตร รายงาน และ dashboard)
FEATURES = {
    "drawdown": "ลงจาก high 52 สัปดาห์ (%)",
    "days_since_high": "วันนับจาก high 52 สัปดาห์",
    "sma200_slope": "ความชัน SMA200 ใน 20 วัน (%)",
    "range_pos": "ตำแหน่งในกรอบ 52 สัปดาห์ (0=ล่างสุด, 1=บนสุด)",
    "off_low": "ห่างจาก low 52 สัปดาห์ (%)",
    "base_days": "วันนับจาก low 52 สัปดาห์ (ความยาวฐาน)",
    "base_width": "ความกว้างฐาน 60 วัน (%)",
    "touches": "จำนวนวันที่แตะขอบล่างฐาน (60 วัน)",
    "higher_low": "low 20 วันล่าสุดเทียบ low 40 วันก่อนหน้า (%)",
    "atr_ratio": "ATR10 / ATR50 (ต่ำ = ผันผวนหด)",
    "bb_width": "ความกว้าง Bollinger 20 วัน (%)",
    "vol_dry": "วอลุ่มเฉลี่ย 20 วัน / 60 วัน (ต่ำ = วอลุ่มแห้ง)",
    "updown_vol": "วอลุ่มวันขึ้น / วันลง ใน 40 วัน (สูง = มีแรงเก็บ)",
    "dist_top": "ห่างจากขอบบนฐาน 40 วัน (%)",
    "vs_sma20": "ราคาเทียบ SMA20 (%)",
    "vs_sma50": "ราคาเทียบ SMA50 (%)",
}
PERCENT_FEATURES = {"drawdown", "sma200_slope", "off_low", "base_width", "higher_low",
                    "bb_width", "dist_top", "vs_sma20", "vs_sma50"}
COUNT_FEATURES = {"days_since_high", "base_days", "touches"}


def _days_since_extreme(values: np.ndarray, window: int, use_max: bool) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) < window:
        return out
    win = sliding_window_view(values, window)
    pos = win.argmax(axis=1) if use_max else win.argmin(axis=1)
    out[window - 1:] = window - 1 - pos
    return out


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """df = OHLCV รายวัน → ตาราง feature รายวัน (index เดียวกัน)"""
    o, h, l, c, v = (df[k] for k in ["Open", "High", "Low", "Close", "Volume"])
    f = pd.DataFrame(index=df.index)

    hi252 = h.rolling(252, min_periods=200).max()
    lo252 = l.rolling(252, min_periods=200).min()
    sma200 = c.rolling(200).mean()
    f["drawdown"] = 1 - c / hi252
    f["days_since_high"] = _days_since_extreme(h.to_numpy(), 252, True)
    f["sma200_slope"] = sma200 / sma200.shift(20) - 1
    f["range_pos"] = (c - lo252) / (hi252 - lo252)
    f["off_low"] = c / lo252 - 1
    f["base_days"] = _days_since_extreme(l.to_numpy(), 252, False)

    hi60, lo60 = h.rolling(60).max(), l.rolling(60).min()
    f["base_width"] = hi60 / lo60 - 1
    f["touches"] = (l <= lo60 * 1.03).astype(float).rolling(60).sum()
    f["higher_low"] = l.rolling(20).min() / l.shift(20).rolling(40).min() - 1

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_ratio"] = tr.rolling(10).mean() / tr.rolling(50).mean()
    sma20 = c.rolling(20).mean()
    f["bb_width"] = 4 * c.rolling(20).std() / sma20

    f["vol_dry"] = v.rolling(20).mean() / v.rolling(60).mean()
    up = c > c.shift()
    up_vol = v.where(up, 0).rolling(40).sum()
    down_vol = v.where(~up, 0).rolling(40).sum()
    f["updown_vol"] = up_vol / down_vol.replace(0, np.nan)

    hi40 = h.rolling(40).max()
    f["dist_top"] = c / hi40 - 1
    f["vs_sma20"] = c / sma20 - 1
    f["vs_sma50"] = c / c.rolling(50).mean() - 1

    # ค่าช่วยสำหรับวาดกราฟ จุด stop และกรองสภาพคล่อง (ไม่ได้ใช้เป็นเงื่อนไขทรง)
    f["base_low"] = lo60
    f["base_top"] = hi40
    f["traded_value"] = (c * v).rolling(20).median()
    return f


def ramp(x: pd.Series, bad: float, good: float) -> pd.Series:
    """0 เมื่อ x อยู่ฝั่ง bad, 1 เมื่อถึง good, เส้นตรงระหว่างกลาง (ใช้ได้ทั้งสองทิศ)"""
    return ((x - bad) / (good - bad)).clip(0, 1)


# เกณฑ์ทรงตั้งต้น: (ชื่อ, น้ำหนัก, ฟังก์ชันให้คะแนน 0–1, คำอธิบาย)
BUILTIN_CRITERIA = [
    ("ขาลงมาก่อน", 2.0, lambda f: ramp(f["drawdown"], 0.20, 0.40), "ลงจาก high 52 สัปดาห์ ≥ 40%"),
    ("ลงมานานพอ", 1.0, lambda f: ramp(f["days_since_high"], 40, 120), "high 52 สัปดาห์ห่างมา ≥ 120 วัน"),
    ("อยู่โซนล่าง", 1.5, lambda f: ramp(f["range_pos"], 0.50, 0.25), "อยู่ล่าง 25% ของกรอบ 52 สัปดาห์"),
    ("หยุดทำ low ใหม่", 1.5, lambda f: ramp(f["base_days"], 10, 30) * ramp(f["base_days"], 220, 160),
     "ไม่ทำ low ใหม่ 30–160 วัน"),
    ("ฐานแคบ", 1.5, lambda f: ramp(f["base_width"], 0.40, 0.20), "กรอบ 60 วันกว้าง ≤ 20%"),
    ("ผันผวนหด", 1.0, lambda f: ramp(f["atr_ratio"], 1.10, 0.75), "ATR10/ATR50 ≤ 0.75"),
    ("วอลุ่มแห้งหรือมีแรงเก็บ", 1.0,
     lambda f: np.maximum(ramp(f["vol_dry"], 1.30, 0.80), ramp(f["updown_vol"], 0.90, 1.30)),
     "วอลุ่ม 20/60 ≤ 0.8 หรือวอลุ่มวันขึ้น/ลง ≥ 1.3"),
    ("low ยกตัว", 1.0, lambda f: ramp(f["higher_low"], -0.03, 0.03), "low 20 วันล่าสุดสูงกว่า low ก่อนหน้า"),
    ("ใกล้ขอบบนฐาน", 1.0, lambda f: ramp(f["dist_top"], -0.15, -0.03), "ราคาห่างขอบบนฐาน ≤ 3%"),
]


def builtin_score(f: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """คืน (คะแนน 0–100, ตารางคะแนนรายเกณฑ์ 0–1)"""
    parts = pd.DataFrame({name: fn(f) for name, _, fn, _ in BUILTIN_CRITERIA}, index=f.index)
    weights = pd.Series({name: w for name, w, _, _ in BUILTIN_CRITERIA})
    score = (parts * weights).sum(axis=1, min_count=len(weights)) / weights.sum() * 100
    # ถ้ายังไม่ได้ลงมาจริง หรือยังไม่อยู่โซนล่าง ไม่นับว่าเป็นทรงนี้ แม้เกณฑ์อื่นผ่าน
    gate = (f["drawdown"] >= 0.20) & (f["range_pos"] <= 0.50)
    return score.where(gate, score * 0.5), parts


def feature_value_text(name: str, value: float) -> str:
    if pd.isna(value):
        return "-"
    if name in PERCENT_FEATURES:
        return f"{value * 100:.1f}%"
    return f"{value:.0f}" if name in COUNT_FEATURES else f"{value:.2f}"
