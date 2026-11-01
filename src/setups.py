"""โหลดและประเมิน set-up จาก setups/*.json

ชนิด set-up:
  builtin  ใช้คะแนนชั้น A (detector.builtin_score) ≥ min_score
  rule     ทุกเงื่อนไขใน conditions ต้องผ่าน (สร้างโดย learn_rule.py หรือเขียนเอง)
  sketch   ทรงที่วาดเองในหน้า app (sketch = ค่า y, length = จำนวนวัน) เรียงตามความเหมือน
  examples มีรูปตัวอย่างแต่ยังไม่มีสูตร เรียงตามความเหมือนกับรูปตัวอย่าง
"""
from __future__ import annotations

import json

import pandas as pd

from common import SETUPS_DIR
from detector import FEATURES, builtin_score, feature_value_text

DEFAULT_SETUP = {
    "name": "base-after-downtrend",
    "title": "หุ้นต้นรอบ",
    "type": "builtin",
    "min_score": 65,
    "description": "นิยามตั้งต้นที่สร้างจากความหมายของทรง: ลงมาแรงและนาน อยู่โซนล่าง หยุดทำ low ใหม่ "
                   "ฐานแคบ ผันผวนหด วอลุ่มแห้ง low ยกตัว และราคาใกล้ขอบบนฐาน ยังไม่ได้เรียนจากตัวอย่าง",
}


def load_setups() -> list[dict]:
    """set-up จากไฟล์ + set-up ที่มีรูปตัวอย่างแล้วแต่ยังไม่ได้สรุปสูตร (ชนิด examples)"""
    from common import load_examples

    SETUPS_DIR.mkdir(exist_ok=True)
    default_path = SETUPS_DIR / f"{DEFAULT_SETUP['name']}.json"
    if not default_path.exists():
        default_path.write_text(json.dumps(DEFAULT_SETUP, indent=2, ensure_ascii=False), encoding="utf-8")
    setups = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(SETUPS_DIR.glob("*.json"))]
    known = {s["name"] for s in setups}
    for name in sorted(set(load_examples()["setup"]) - known):
        setups.append({"name": name, "title": name, "type": "examples",
                       "description": "มีรูปตัวอย่างแล้ว แต่ยังไม่ครบ 3 รูปหรือยังไม่ได้รัน learn_rule.py "
                                      "จึงจัดอันดับจากความเหมือนกับรูปตัวอย่างอย่างเดียว"})
    return setups


def condition_mask(f: pd.DataFrame, cond: dict) -> pd.Series:
    x = f[cond["feature"]]
    return x <= cond["value"] if cond["op"] == "<=" else x >= cond["value"]


def condition_text(cond: dict) -> str:
    return f"{FEATURES[cond['feature']]} {cond['op']} {feature_value_text(cond['feature'], cond['value'])}"


def evaluate(setup: dict, f: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """คืน (ผ่าน set-up หรือไม่, คะแนน 0–100) ของทุกแถวใน f"""
    score, _ = builtin_score(f)
    if setup["type"] == "builtin":
        return score >= setup.get("min_score", 65), score
    if setup["type"] in ("examples", "sketch"):   # ยังไม่มีสูตร: จัดอันดับด้วยความเหมือนอย่างเดียว
        return pd.Series(False, index=f.index), score
    passed = pd.Series(True, index=f.index)
    for cond in setup["conditions"]:
        passed &= condition_mask(f, cond)
    return passed, score


def reasons(setup: dict, row: pd.Series, parts: pd.Series | None = None) -> list[dict]:
    """เหตุผลรายข้อว่าหุ้นแถวนี้ผ่าน/ไม่ผ่านเงื่อนไขไหน สำหรับแสดงใน dashboard"""
    if setup["type"] == "builtin":
        from detector import BUILTIN_CRITERIA
        return [{"text": f"{name}: {desc}", "ok": bool(parts[name] >= 0.7), "level": round(float(parts[name]), 2)}
                for name, _, _, desc in BUILTIN_CRITERIA]
    out = []
    for cond in setup.get("conditions", []):
        v = row[cond["feature"]]
        ok = bool(v <= cond["value"]) if cond["op"] == "<=" else bool(v >= cond["value"])
        out.append({"text": f"{condition_text(cond)} (ตอนนี้ {feature_value_text(cond['feature'], v)})",
                    "ok": ok, "level": 1.0 if ok else 0.0})
    return out
