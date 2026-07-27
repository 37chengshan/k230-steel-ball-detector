#!/usr/bin/env python3
"""Live, dependency-free local dashboard for steel-ball training runs."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "高清 1024": ROOT / "training" / "runs" / "detect" / "steel_ball_reference_yolo11n_1024",
    "快速 640": ROOT / "training" / "runs" / "detect" / "steel_ball_reference_yolo11n_640_fast",
    "YOLO26 1024": ROOT / "training" / "runs" / "detect" / "steel_ball_reference_yolo26n_1024_live",
}


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0").strip())
    except ValueError:
        return 0.0


def tail(path: Path, limit: int = 9000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - limit))
        # Keep carriage returns here: Rich/TQDM use them to redraw one progress
        # line, and terminal_view() turns them into readable line breaks.
        return handle.read().decode("utf-8", errors="replace")


def terminal_view(logs: str, synthetic: str) -> str:
    """Turn Rich's animated progress stream into a stable, readable web log."""
    if not logs:
        return synthetic
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", logs).replace("\r", "\n")
    text = re.sub(r"\s+\d+%\|[^|]*\|\s*(\d+/\d+)", r"  batch=\1", text)
    text = re.sub(r"[^\x09\x0a\x0d\x20-\x7e]", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    unique: list[str] = []
    for line in lines:
        if not unique or unique[-1] != line:
            unique.append(line)
    return "\n".join(unique[-10:]) or synthetic


def read_run(name: str, directory: Path) -> dict:
    results = directory / "results.csv"
    if not results.is_file():
        logs = tail(directory / "train.err.log") or tail(directory / "train.out.log")
        return {"name": name, "epochs": [], "map50": [], "precision": [], "recall": [], "terminal": terminal_view(logs, "等待训练进程写入第一轮数据…"), "state": "启动中", "updated": "—"}
    with results.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = [int(number(row, "epoch")) + 1 for row in rows]
    values = {
        "map50": [number(row, "metrics/mAP50(B)") for row in rows],
        "precision": [number(row, "metrics/precision(B)") for row in rows],
        "recall": [number(row, "metrics/recall(B)") for row in rows],
    }
    logs = tail(directory / "train.err.log") or tail(directory / "train.out.log")
    last = rows[-1] if rows else {}
    synthetic = "[monitor] %s\n[epoch] %d completed\n[metrics] mAP50=%.4f  precision=%.4f  recall=%.4f\n[status] waiting for next epoch output…" % (
        datetime.now().strftime("%H:%M:%S"),
        epochs[-1] if epochs else 0,
        number(last, "metrics/mAP50(B)"), number(last, "metrics/precision(B)"), number(last, "metrics/recall(B)"),
    )
    return {
        "name": name,
        "epochs": epochs,
        **values,
        "terminal": terminal_view(logs, synthetic),
        "state": "训练中/已完成",
        "updated": datetime.fromtimestamp(results.stat().st_mtime).strftime("%H:%M:%S"),
    }


HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>钢珠训练台</title><style>
:root{--sun:#f1d58b;--paper:#fff7e4;--ink:#34241c;--quiet:#866b56;--rule:#d8bd78;--orange:#da7755;--orange-deep:#a6452b;--orange-pale:#f7c7ae;--green:#729c83;--gold:#d7a22c;--screen:#28221d;--screen-ink:#f9d8a4}*{box-sizing:border-box}body{margin:0;background:var(--sun);color:var(--ink);font-family:"Microsoft YaHei",ui-sans-serif,system-ui,sans-serif}header{max-width:1680px;margin:auto;padding:28px 32px 23px;display:flex;align-items:end;justify-content:space-between;gap:24px;border-bottom:2px solid var(--ink)}.kicker{font:700 11px/1.2 ui-monospace,Consolas,monospace;letter-spacing:.12em;color:var(--orange-deep);text-transform:uppercase}h1{margin:7px 0 0;font-family:Georgia,"Songti SC",serif;font-size:35px;font-weight:600;letter-spacing:0}header p{margin:6px 0 0;color:var(--quiet);font-size:14px}.live{display:flex;align-items:center;gap:8px;font:12px ui-monospace,Consolas,monospace;color:var(--orange-deep);white-space:nowrap}.dot{width:9px;height:9px;background:var(--orange);border-radius:50%;box-shadow:0 0 0 4px var(--orange-pale);animation:blink 1.5s step-end infinite}@keyframes blink{50%{opacity:.3}}main{max-width:1680px;margin:auto;padding:22px 32px 32px}.guide{display:flex;align-items:center;justify-content:space-between;margin:0 0 14px;color:var(--quiet);font-size:12px}.legend{display:flex;gap:16px}.legend i{display:inline-block;width:17px;height:2px;vertical-align:middle;margin-right:5px}.runs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.run{position:relative;min-width:0;background:var(--paper);border:1px solid var(--ink);padding:19px}.run:before{content:"";position:absolute;left:-1px;top:-1px;width:75px;height:6px;background:var(--orange)}.run-head{display:flex;justify-content:space-between;align-items:start;gap:10px;padding:7px 0 16px;border-bottom:1px solid var(--rule)}.run h2{margin:0;font:600 23px/1 Georgia,"Songti SC",serif}.run-head small{color:var(--quiet);font-size:11px;text-align:right;line-height:1.55}.metrics{display:grid;grid-template-columns:repeat(3,1fr);border-bottom:1px solid var(--rule)}.metric{padding:15px 8px 13px 0}.metric+.metric{padding-left:10px;border-left:1px solid var(--rule)}.metric span{display:block;color:var(--quiet);font:11px ui-monospace,Consolas,monospace}.metric b{display:block;margin-top:4px;color:var(--orange-deep);font:600 25px/1 Georgia,serif}.chart-wrap{padding:17px 0 14px}.chart{display:block;width:100%;height:230px;background:#fffaf0;border:1px solid #e7d19a}.terminal-label{display:flex;justify-content:space-between;align-items:center;border-top:1px solid var(--rule);padding:13px 0 7px;color:var(--quiet);font:11px ui-monospace,Consolas,monospace}.terminal{height:204px;overflow:auto;margin:0;padding:13px 14px;background:var(--screen);color:var(--screen-ink);border-left:4px solid var(--orange);font:11px/1.58 ui-monospace,Consolas,monospace;white-space:pre-wrap;word-break:break-word}.terminal::-webkit-scrollbar{width:8px}.terminal::-webkit-scrollbar-thumb{background:#725c49}@media(max-width:1180px){.runs{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:860px){header{padding:21px 18px;align-items:start;flex-direction:column}h1{font-size:30px}main{padding:16px 18px}.runs{grid-template-columns:1fr}.guide{align-items:start;gap:12px;flex-direction:column}.chart{height:210px}}@media(prefers-reduced-motion:reduce){.dot{animation:none}}</style></head><body><header><div><div class="kicker">K230 / Steel ball detector / model comparison</div><h1>钢珠训练台</h1><p>YOLO11 高清、快速版与 YOLO26 同屏比较</p></div><div class="live"><span class="dot"></span><span id="clock">连接中</span></div></header><main><div class="guide"><span>三条训练在同一屏，终端与曲线每 2 秒刷新。</span><div class="legend"><span><i style="background:#da7755"></i>mAP50</span><span><i style="background:#729c83"></i>精确率</span><span><i style="background:#d7a22c"></i>召回率</span></div></div><div class="runs" id="runs"></div></main><script>
const esc=s=>String(s).replace(/[&<>]/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[x]));const val=a=>a.length?a[a.length-1]:0;
function panel(d){return `<section class="run"><div class="run-head"><h2>${esc(d.name)}</h2><small>${d.state}<br>更新 ${d.updated} · ${d.epochs.length} 轮</small></div><div class="metrics"><div class="metric"><span>mAP50</span><b>${val(d.map50).toFixed(3)}</b></div><div class="metric"><span>精确率</span><b>${val(d.precision).toFixed(3)}</b></div><div class="metric"><span>召回率</span><b>${val(d.recall).toFixed(3)}</b></div></div><div class="chart-wrap"><canvas class="chart" width="620" height="230" id="chart-${d.name}"></canvas></div><div class="terminal-label"><span>LIVE OUTPUT</span><span>每 2 秒刷新</span></div><pre class="terminal" id="term-${d.name}">${esc(d.terminal)}</pre></section>`}
function line(x,a,color,w,h,p){if(!a.length)return;x.strokeStyle=color;x.lineWidth=2;x.beginPath();a.forEach((n,i)=>{const px=p+i*(w-p*2)/Math.max(1,a.length-1),py=h-p-n*(h-p*2);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()}function chart(id,d){const c=document.getElementById('chart-'+id),x=c.getContext('2d'),w=c.width,h=c.height,p=30;x.clearRect(0,0,w,h);x.strokeStyle='#ead7a4';x.lineWidth=1;x.fillStyle='#9c806d';x.font='10px ui-monospace,monospace';for(let i=0;i<5;i++){const y=p+i*(h-p*2)/4;x.beginPath();x.moveTo(p,y);x.lineTo(w-p,y);x.stroke();x.fillText((1-i/4).toFixed(2),2,y+3)}line(x,d.map50,'#da7755',w,h,p);line(x,d.precision,'#729c83',w,h,p);line(x,d.recall,'#d7a22c',w,h,p);x.fillText('epoch',w-55,h-8)}
async function refresh(){try{const all=await fetch('/api?'+Date.now()).then(r=>r.json()),runs=Object.values(all),host=document.querySelector('#runs');host.innerHTML=runs.map(panel).join('');runs.forEach(d=>{chart(d.name,d);const t=document.getElementById('term-'+d.name);t.scrollTop=t.scrollHeight});document.querySelector('#clock').textContent='实时更新 '+new Date().toLocaleTimeString()}catch(e){document.querySelector('#clock').textContent='等待训练服务'}}refresh();setInterval(refresh,2000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith("/api"):
            body = json.dumps({name: read_run(name, directory) for name, directory in RUNS.items()}, ensure_ascii=False).encode("utf-8")
            content_type = "application/json; charset=utf-8"
        else:
            body = HTML.encode("utf-8")
            content_type = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("TRAINING_DASHBOARD=http://127.0.0.1:8765")
    server.serve_forever()
