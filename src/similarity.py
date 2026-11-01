"""ชั้น B: หาหุ้นที่ทรงกราฟคล้ายรูปตัวอย่าง

ตัวอย่างหนึ่งรูป = หุ้นหนึ่งตัว + ช่วงวันที่ฐานเริ่ม (base_start) + วันเข้า (entry_date)
วันสัญญาณ = แท่งสุดท้ายก่อนวันเข้า เพราะสัญญาณเกิดตอนปิดตลาด แล้วเข้าวันถัดไป
ช่วงที่ใช้เทียบ = ขาลงก่อนฐาน (ยาวเท่าฐาน อย่างน้อย 20 วัน) + ฐาน จนถึงวันสัญญาณ
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from detector import FEATURES

PATH_LEN = 64          # ทุกช่วงถูกย่อ/ขยายให้ยาวเท่านี้ก่อนเทียบ
BAND = 8               # DTW ยืดหดเวลาได้ไม่เกิน 8 จุดจาก 64
SCALES = (0.75, 1.0, 1.33)   # ลองช่วงสั้นกว่า/ยาวกว่าตัวอย่าง เพราะฐานแต่ละตัวยาวไม่เท่ากัน
WEIGHTS = {"shape": 0.5, "volume": 0.2, "features": 0.3}


@dataclass
class Template:
    ticker: str
    label: str
    signal_date: pd.Timestamp
    length: int
    price_path: np.ndarray
    volume_path: np.ndarray
    features: pd.Series
    chart: pd.DataFrame


def resample(values: np.ndarray, n: int = PATH_LEN) -> np.ndarray:
    x_old = np.linspace(0, 1, len(values))
    return np.interp(np.linspace(0, 1, n), x_old, values)


def zscore(values: np.ndarray) -> np.ndarray:
    sd = values.std()
    return (values - values.mean()) / (sd if sd > 1e-12 else 1.0)


def price_shape(close: np.ndarray) -> np.ndarray:
    return zscore(resample(np.log(close)))


def volume_shape(volume: np.ndarray) -> np.ndarray:
    v = pd.Series(np.log1p(volume)).rolling(5, min_periods=1).mean().to_numpy()
    return zscore(resample(v))


def dtw_batch(template: np.ndarray, candidates: np.ndarray, band: int = BAND) -> np.ndarray:
    """DTW ระหว่าง template (n,) กับ candidates (N, n) พร้อมกันทุกตัว คืนระยะเฉลี่ยต่อจุด (N,)"""
    n = len(template)
    big = np.inf
    cost = np.full((n + 1, n + 1, len(candidates)), big)
    cost[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(n, i + band) + 1):
            d = np.abs(template[i - 1] - candidates[:, j - 1])
            cost[i, j] = d + np.minimum(np.minimum(cost[i - 1, j], cost[i, j - 1]), cost[i - 1, j - 1])
    return cost[n, n] / n


def signal_index(df: pd.DataFrame, entry_date: str) -> int:
    idx = df.index.searchsorted(pd.Timestamp(entry_date)) - 1
    if idx < 0:
        raise ValueError(f"ไม่มีข้อมูลราคาก่อนวันเข้า {entry_date}")
    return int(idx)


def build_template(ticker: str, label: str, df: pd.DataFrame, feats: pd.DataFrame,
                   base_start: str, entry_date: str) -> Template:
    sig = signal_index(df, entry_date)
    start = int(df.index.searchsorted(pd.Timestamp(base_start)))
    base_len = max(sig - start, 10)
    first = max(0, start - max(base_len, 20))
    window = df.iloc[first:sig + 1]
    chart_first = max(0, first - 20)
    return Template(
        ticker=ticker, label=label, signal_date=df.index[sig], length=len(window),
        price_path=price_shape(window["Close"].to_numpy()),
        volume_path=volume_shape(window["Volume"].to_numpy()),
        features=feats.iloc[sig][list(FEATURES)],
        chart=df.iloc[chart_first:min(len(df), sig + 41)],
    )


def candidate_paths(prices: dict[str, pd.DataFrame], end_idx: dict[str, int], length: int):
    """ตัดช่วงยาว length ที่จบวันสแกนของทุกหุ้น คืน (symbols, price_paths, volume_paths)"""
    symbols, pp, vp = [], [], []
    for s, df in prices.items():
        e = end_idx[s]
        if e + 1 < length:
            continue
        w = df.iloc[e + 1 - length:e + 1]
        symbols.append(s)
        pp.append(price_shape(w["Close"].to_numpy()))
        vp.append(volume_shape(w["Volume"].to_numpy()))
    return symbols, np.array(pp), np.array(vp)


def feature_scale(feature_rows: pd.DataFrame) -> pd.Series:
    """ความกระจายของแต่ละตัววัดในตลาด (ใช้ MAD กันค่าสุดโต่ง) ไว้แปลงระยะห่างให้เทียบกันได้"""
    x = feature_rows[list(FEATURES)]
    mad = (x - x.median()).abs().median() * 1.4826
    return mad.replace(0, np.nan).fillna(x.std()).fillna(1.0)


def similarity_scores(template: Template, prices: dict[str, pd.DataFrame], end_idx: dict[str, int],
                      today_feats: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    """คะแนนความเหมือน 0–100 ของทุกหุ้นกับตัวอย่างหนึ่งรูป แยกเป็นทรงราคา วอลุ่ม และตัววัด"""
    best = {}
    for k in SCALES:
        length = max(30, int(round(template.length * k)))
        symbols, pp, vp = candidate_paths(prices, end_idx, length)
        if not symbols:
            continue
        d_shape = dtw_batch(template.price_path, pp)
        d_vol = np.abs(vp - template.volume_path).mean(axis=1)
        for s, ds, dv in zip(symbols, d_shape, d_vol):
            if s not in best or ds < best[s][0]:
                best[s] = (ds, dv, length)
    if not best:
        return pd.DataFrame()
    out = pd.DataFrame.from_dict(best, orient="index", columns=["d_shape", "d_volume", "window"])
    z = ((today_feats.loc[out.index, list(FEATURES)] - template.features) / scale).abs()
    out["d_features"] = z.mean(axis=1, skipna=True)
    out["sim_shape"] = 100 * np.exp(-1.5 * out["d_shape"])
    out["sim_volume"] = 100 * np.exp(-1.5 * out["d_volume"])
    out["sim_features"] = 100 * np.exp(-0.6 * out["d_features"].fillna(3))
    out["similarity"] = sum(out[f"sim_{k}"] * w for k, w in WEIGHTS.items())
    return out
