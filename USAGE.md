---
title: วิธีใช้ Setup Screener
type: instruction
created: 2026-11-01
updated: 2026-11-01
tags: [screener, setup, how-to]
sources: [Yahoo Finance via yfinance]
status: draft
---

# วิธีใช้ Setup Screener

## ติดตั้งครั้งแรก

```bash
python -m pip install -r requirements.txt
python src/universe.py        # รายชื่อหุ้น SET100 + S&P 500
python src/fetch_prices.py    # ราคา 5 ปี (~2 นาที) เก็บไว้ที่ data/prices/ ไม่ขึ้น git
```

## ใช้ทุกวัน

```bash
python src/fetch_prices.py    # อัปเดตราคา
python src/backtest.py        # ทดสอบย้อนหลังทุก set-up (~20 วินาที)
python src/scan.py            # สแกน → output/dashboard.html และ output/signals.csv
```

เปิด `output/dashboard.html` ในเบราว์เซอร์ เลือก set-up และตลาด แล้วกดที่หุ้นเพื่อดูกราฟพร้อมกรอบฐานและเหตุผลรายข้อ

## สอนทรงใหม่จากรูปกราฟ

1. แคปกราฟที่เห็นชื่อหุ้นและแกนวันที่ (เช่น TradingView) วางไว้ใน `examples/inbox/` แล้วบอก Claude ว่า "เพิ่มตัวอย่าง ทรง <ชื่อทรง>"
2. Claude อ่านรูปแล้วถามยืนยัน ticker, ตลาด, วันที่ฐานเริ่ม และวันเข้า จากนั้นรัน
   `python src/examples.py add --setup <ชื่อทรง> --ticker ... --market SET --base-start YYYY-MM-DD --entry YYYY-MM-DD --label good --image examples/inbox/<รูป>`
3. ตั้งแต่ 1 รูป: `python src/scan.py` จะมี set-up ชื่อนั้นที่เรียงหุ้นตามความเหมือนกับรูปตัวอย่าง (ทรงราคา + วอลุ่ม + ตัววัด)
4. ตั้งแต่ 3 รูป: `python src/learn_rule.py --setup <ชื่อทรง> --title "ชื่อที่อ่านง่าย"` สรุปเป็นสูตร → `setups/<ชื่อทรง>.json` และ `output/rule_report_<ชื่อทรง>.md`
5. ในหน้า dashboard กด ✓ ใช่ / ✗ ไม่ใช่ แล้วกด "ดาวน์โหลด feedback.csv" จากนั้นรัน
   `python src/examples.py import <ไฟล์ feedback.csv>` แล้วรัน learn_rule และ scan ใหม่ ตัวที่ติ๊ก "ไม่ใช่" จะถูกใช้ตัดหุ้นหน้าตาแบบนั้นออก

แก้สูตรเองได้ใน `setups/<ชื่อทรง>.json` (ค่า `value` ของแต่ละเงื่อนไข)

## ตรวจว่าระบบไม่โกงตัวเอง

```bash
python src/selftest.py
```

ตรวจว่าตัววัดไม่ใช้ข้อมูลอนาคต และใช้หุ้นเป็นรูปตัวอย่างแล้วสแกนย้อนไปวันนั้นต้องเจอหุ้นตัวเองเป็นอันดับ 1

## ไฟล์

| ไฟล์ | ทำอะไร |
|---|---|
| `src/detector.py` | ชั้น A: แปลงกราฟเป็นตัววัด 16 ตัว + คะแนนทรงตั้งต้น |
| `src/similarity.py` | ชั้น B: เทียบทรงราคา (DTW) วอลุ่ม และตัววัด กับรูปตัวอย่าง |
| `src/learn_rule.py` | ชั้น C: สรุปรูปตัวอย่างเป็นสูตรสแกน |
| `src/scan.py` | สแกนทุก set-up และสร้าง dashboard |
| `src/backtest.py` | ทดสอบย้อนหลัง เทียบกับซื้อแบบไม่เลือก |
| `src/examples.py` | เพิ่มรูปตัวอย่าง และนำเข้าป้ายจาก dashboard |
| `setups/*.json` | สูตรของแต่ละ set-up |
| `examples/examples.csv` | ทะเบียนรูปตัวอย่าง |
