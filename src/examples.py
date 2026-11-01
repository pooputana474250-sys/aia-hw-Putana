"""จัดการรูปกราฟตัวอย่าง → examples/examples.csv

เพิ่มตัวอย่างหนึ่งรูป (หลังยืนยัน ticker และช่วงวันจากรูปแล้ว):
  python src/examples.py add --setup my-base --ticker PTT --market SET \
      --base-start 2025-03-10 --entry 2025-06-02 --label good --image examples/inbox/ptt.png

นำเข้าป้าย ใช่/ไม่ใช่ ที่กดใน dashboard (ไฟล์ที่ดาวน์โหลดจากปุ่ม "ดาวน์โหลด feedback.csv"):
  python src/examples.py import path/to/feedback.csv

ดูรายการ:
  python src/examples.py list
"""
from __future__ import annotations

import argparse

import pandas as pd

from common import EXAMPLES_CSV, EXAMPLES_DIR, load_examples


def save(df: pd.DataFrame) -> None:
    EXAMPLES_DIR.mkdir(exist_ok=True)
    df = df.drop_duplicates(subset=["setup", "ticker", "market", "entry_date"], keep="last")
    df.sort_values(["setup", "entry_date"]).to_csv(EXAMPLES_CSV, index=False, encoding="utf-8")


def add(row: dict) -> None:
    for k in ("base_start", "entry_date"):
        pd.Timestamp(row[k])  # ตรวจรูปแบบวันที่ ผิดจะ error ทันที
    if row["label"] not in ("good", "bad"):
        raise SystemExit("label ต้องเป็น good หรือ bad")
    row["ticker"] = row["ticker"].upper().removesuffix(".BK")
    save(pd.concat([load_examples(), pd.DataFrame([row])], ignore_index=True))
    print(f"เพิ่ม {row['label']}: {row['ticker']} ({row['market']}) ฐานเริ่ม {row['base_start']} เข้า {row['entry_date']}")


def import_feedback(path: str) -> None:
    fb = pd.read_csv(path, dtype=str).fillna("")
    fb["image"] = ""
    fb["note"] = fb.get("note", "dashboard feedback")
    save(pd.concat([load_examples(), fb[load_examples().columns]], ignore_index=True))
    print(f"นำเข้า {len(fb)} ป้าย (good {sum(fb['label'] == 'good')}, bad {sum(fb['label'] == 'bad')})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("--setup", required=True)
    a.add_argument("--ticker", required=True)
    a.add_argument("--market", required=True, choices=["SET", "US"])
    a.add_argument("--base-start", required=True)
    a.add_argument("--entry", required=True)
    a.add_argument("--label", default="good", choices=["good", "bad"])
    a.add_argument("--image", default="")
    a.add_argument("--note", default="")
    i = sub.add_parser("import")
    i.add_argument("path")
    sub.add_parser("list")
    args = ap.parse_args()

    if args.cmd == "add":
        add({"setup": args.setup, "ticker": args.ticker, "market": args.market,
             "base_start": args.base_start, "entry_date": args.entry, "label": args.label,
             "image": args.image, "note": args.note})
    elif args.cmd == "import":
        import_feedback(args.path)
    else:
        print(load_examples().to_string(index=False))


if __name__ == "__main__":
    main()
