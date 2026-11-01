"""สแกนใหม่ประจำวัน: อัปเดตราคา → ทดสอบย้อนหลัง → สแกน (Windows Task Scheduler เรียกทุกวัน 08:00)

เวลา 08:00 ไทย ได้ราคาปิดหุ้นสหรัฐของคืนก่อน และราคาปิดหุ้นไทยของวันทำการก่อน ก่อนตลาดไทยเปิด 10:00
บันทึกผลทุกครั้งไว้ที่ output/daily.log (ไม่ขึ้น git)

ใช้เอง:  python src/daily.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

from common import OUTPUT_DIR, ROOT

STEPS = [["src/fetch_prices.py"], ["src/backtest.py"], ["src/scan.py"]]
LOG = OUTPUT_DIR / "daily.log"


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {datetime.now():%Y-%m-%d %H:%M} ===\n")
        for step in STEPS:
            r = subprocess.run([sys.executable, "-W", "ignore", *step], cwd=ROOT, capture_output=True,
                               text=True, encoding="utf-8", errors="replace",
                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            tail = "\n".join((r.stdout or "").strip().splitlines()[-4:])
            log.write(f"[{'ok' if r.returncode == 0 else 'FAIL'}] {step[0]}\n{tail}\n")
            if r.returncode != 0:
                log.write((r.stderr or "")[-2000:] + "\n")
                return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
