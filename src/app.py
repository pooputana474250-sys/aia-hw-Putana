"""หน้าเว็บในเครื่องสำหรับใส่ทรงกราฟแล้ววิเคราะห์ทันที

ใช้:  python src/app.py   แล้วเปิด http://127.0.0.1:8770

ทำได้ 3 ทาง
1. วาดทรง: ลากเส้นทรงที่ชอบ → หาหุ้นที่ราคาช่วงล่าสุดหน้าตาเหมือน → บันทึกเป็น set-up ได้
2. เลือกจากกราฟจริง: พิมพ์ชื่อหุ้น ลากเลือกช่วงฐานบนกราฟ → เพิ่มเป็นรูปตัวอย่าง → หาหุ้นที่คล้าย
   ครบ 3 ตัวอย่างกด "สรุปเป็นสูตร" ได้
3. อัปโหลดรูปกราฟ: เก็บรูปไว้ใน examples/inbox/ แล้วกรอกชื่อหุ้นและช่วงวันข้างรูป (หรือให้ Claude อ่านรูปให้)

ใช้เฉพาะในเครื่องตัวเอง (ผูกกับ 127.0.0.1) ไม่เปิดให้เครื่องอื่นเข้า
"""
from __future__ import annotations

import base64
import json
import re
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from common import EXAMPLES_DIR, PRICE_META, ROOT, SETUPS_DIR, load_examples, price_meta, to_symbol
from examples import add as add_example
from market import load_market
from scan import bars, latest_date, prepare, scan_setup
from setups import load_setups

HOST, PORT = "127.0.0.1", 8770
INBOX = EXAMPLES_DIR / "inbox"

STOCKS: dict = {}
SCAN_DATE = END_IDX = TODAY = SCALE = None
LOADED_AT = -1.0


def refresh_if_stale() -> None:
    """โหลดราคาใหม่เมื่อ fetch_prices (เช่นงานสแกน 08:00) อัปเดตข้อมูล ไม่ต้องปิดเปิดหน้าเว็บ"""
    global STOCKS, SCAN_DATE, END_IDX, TODAY, SCALE, LOADED_AT
    stamp = PRICE_META.stat().st_mtime if PRICE_META.exists() else 0.0
    if stamp == LOADED_AT:
        return
    STOCKS = load_market()
    SCAN_DATE = latest_date(STOCKS)
    END_IDX, TODAY, SCALE = prepare(STOCKS, SCAN_DATE)
    LOADED_AT = stamp
    print(f"  โหลดราคาใหม่ (ข้อมูลถึง {SCAN_DATE.date()})")


def slug(text: str) -> str:
    s = re.sub(r"[^\w\-]+", "-", text.strip().lower()).strip("-")
    if not s:
        raise ValueError("ตั้งชื่อ set-up ก่อน")
    return s


def setup_by_name(name: str) -> dict:
    for s in load_setups():
        if s["name"] == name:
            return s
    raise ValueError(f"ไม่เจอ set-up {name}")


def run_setup(setup: dict) -> dict:
    signals, examples = scan_setup(setup, STOCKS, END_IDX, TODAY, SCALE, SCAN_DATE)
    return {"setup": setup, "signals": signals, "examples": examples}


def api_state(_: dict) -> dict:
    meta = price_meta()
    retrieved = sorted({v["retrieved"] for v in meta.values()})
    ex = load_examples()
    return {
        "scan_date": SCAN_DATE.strftime("%Y-%m-%d"),
        "retrieved": retrieved[-1] if retrieved else "",
        "tickers": sorted(f"{s.ticker}|{s.market}" for s in STOCKS.values()),
        "setups": [{"name": s["name"], "title": s.get("title", s["name"]), "type": s["type"],
                    "good": int(((ex["setup"] == s["name"]) & (ex["label"] == "good")).sum()),
                    "bad": int(((ex["setup"] == s["name"]) & (ex["label"] == "bad")).sum())}
                   for s in load_setups()],
    }


def api_chart(q: dict) -> dict:
    s = STOCKS.get(to_symbol(q["ticker"], q["market"]))
    if s is None:
        raise ValueError(f"ไม่มีราคาของ {q['ticker']} ({q['market']}) ถ้าเป็นหุ้นนอกรายชื่อ ให้เพิ่มใน universe แล้วรัน fetch_prices")
    days = int(q.get("days", 500))
    # ช่วงยาว (5–10 ปี) แสดงเป็นแท่งรายสัปดาห์ เลือกฐานได้ละเอียดระดับสัปดาห์
    return {"ticker": s.ticker, "market": s.market, "name": s.name, "tv": s.tv,
            "chart": bars(s.prices.iloc[-days:], max_bars=600)}


def api_sketch(q: dict) -> dict:
    points, length = q["points"], int(q["length"])
    if len(points) < 5:
        raise ValueError("วาดเส้นให้ยาวกว่านี้")
    setup = {"name": "_sketch", "title": "ทรงที่วาด", "type": "sketch", "sketch": points, "length": length}
    return run_setup(setup)


def api_save_sketch(q: dict) -> dict:
    name = slug(q["name"])
    setup = {"name": name, "title": q.get("title") or q["name"], "type": "sketch",
             "sketch": [round(float(p), 4) for p in q["points"]], "length": int(q["length"]),
             "created": datetime.now().strftime("%Y-%m-%d"),
             "description": f"ทรงที่วาดเองในหน้า app ยาวประมาณ {int(q['length'])} วัน เรียงหุ้นตามความเหมือนของทรงราคา"}
    SETUPS_DIR.mkdir(exist_ok=True)
    (SETUPS_DIR / f"{name}.json").write_text(json.dumps(setup, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"saved": f"setups/{name}.json", "name": name}


def api_add_example(q: dict) -> dict:
    name = slug(q["setup"])
    add_example({"setup": name, "ticker": q["ticker"], "market": q["market"], "base_start": q["base_start"],
                 "entry_date": q["entry_date"], "label": q.get("label", "good"),
                 "image": q.get("image", ""), "note": q.get("note", "app")})
    return run_setup(setup_by_name(name))


def api_run(q: dict) -> dict:
    return run_setup(setup_by_name(q["setup"]))


def api_learn(q: dict) -> dict:
    from learn_rule import learn
    try:
        setup = learn(q["setup"], q.get("title"), STOCKS)
    except SystemExit as e:
        raise ValueError(str(e)) from None
    report = ROOT / "output" / f"rule_report_{q['setup']}.md"
    return {**run_setup(setup), "report": report.read_text(encoding="utf-8") if report.exists() else ""}


def api_upload(q: dict) -> dict:
    m = re.match(r"data:image/(png|jpe?g|webp);base64,(.+)", q["data"], re.S)
    if not m:
        raise ValueError("รับเฉพาะรูป png / jpg / webp")
    INBOX.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^\w\-]+", "_", q.get("filename", "chart").rsplit(".", 1)[0])[:40]
    path = INBOX / f"{datetime.now():%Y-%m-%d_%H%M%S}_{stem}.{m.group(1).replace('jpeg', 'jpg')}"
    path.write_bytes(base64.b64decode(m.group(2)))
    return {"image": path.relative_to(ROOT).as_posix()}


ROUTES = {"/api/state": api_state, "/api/chart": api_chart, "/api/sketch": api_sketch,
          "/api/save_sketch": api_save_sketch, "/api/example": api_add_example, "/api/run": api_run,
          "/api/learn": api_learn, "/api/upload": api_upload}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _handle(self, q: dict) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            html = (ROOT / "src" / "app_template.html").read_text(encoding="utf-8")
            return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        if path == "/dashboard":   # ตารางสรุปจากงานสแกน 08:00 (output/dashboard.html)
            page = ROOT / "output" / "dashboard.html"
            if not page.exists():
                return self._json(404, {"error": "ยังไม่มี dashboard รัน python src/scan.py ก่อน"})
            return self._send(200, page.read_bytes(), "text/html; charset=utf-8")
        fn = ROUTES.get(path)
        if fn is None:
            return self._json(404, {"error": "ไม่มีหน้านี้"})
        try:
            refresh_if_stale()
            self._json(200, fn(q))
        except (ValueError, KeyError) as e:
            self._json(400, {"error": str(e)})
        except Exception as e:  # แสดงข้อผิดพลาดในหน้าเว็บ แทนที่จะให้เซิร์ฟเวอร์ล้ม
            traceback.print_exc()
            self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self) -> None:
        self._handle({k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()})

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length", 0))
        self._handle(json.loads(self.rfile.read(n) or b"{}"))

    def log_message(self, fmt: str, *args) -> None:
        if "/api/" in (args[0] if args else ""):
            print(f"  {args[0]}")


if __name__ == "__main__":
    refresh_if_stale()
    print(f"เปิด http://{HOST}:{PORT}  (วันสแกน {SCAN_DATE.date()} · กด Ctrl+C เพื่อปิด)")
    HTTPServer((HOST, PORT), Handler).serve_forever()
