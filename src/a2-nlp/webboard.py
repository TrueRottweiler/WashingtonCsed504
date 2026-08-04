"""
webboard.py -- a browser dashboard for training runs, reachable from any machine.

The terminal dashboard is fine when you are sitting at the workstation. A run that lasts two
days is not watched that way: you want to check it from a laptop in another room or a phone at
midnight, and you want curves rather than a sparkline made of block characters.

So this serves one self-contained page over HTTP. No build step, no CDN, no external requests --
the page is a few hundred lines of inline SVG and fetch(), and everything it draws comes from
the same files the terminal dashboard reads: runs/*.jsonl for the curves, runs/*_result.json for
finished runs, logs/*.log for what is alive, and nvidia-smi for the cards.

    python webboard.py                 # http://localhost:8770
    python webboard.py --port 9000 --host 0.0.0.0     # reachable from other machines

--host 0.0.0.0 binds every interface, so anything on your network can read it. That is the point
-- and it is also the caveat, since there is no authentication. Use it on a network you trust.

Read-only. It never writes to runs/ and never touches a training process.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import math
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERVER_STARTED = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
LOGS = os.path.join(HERE, 'logs')

# nvidia-smi costs ~50 ms, and several browsers polling at once should not multiply that.
STUDY_TAG = re.compile(r'^[a-z]+_[\d.]+[kM]?_[\d.]+[kM]?(_[a-z]+)?_s\d+$')

_GPU_CACHE = {'t': 0.0, 'data': []}
_GPU_LOCK = threading.Lock()


def gpus() -> list[dict]:
    with _GPU_LOCK:
        if time.time() - _GPU_CACHE['t'] < 1.5:
            return _GPU_CACHE['data']
        out = []
        try:
            q = ('index,name,utilization.gpu,memory.used,memory.total,'
                 'power.draw,power.limit,temperature.gpu')
            raw = subprocess.run(['nvidia-smi', f'--query-gpu={q}',
                                  '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=5).stdout
            for line in raw.strip().splitlines():
                f = [x.strip() for x in line.split(',')]
                out.append({'i': int(f[0]), 'name': f[1], 'util': float(f[2]),
                            'mem': float(f[3]) / 1024, 'mem_tot': float(f[4]) / 1024,
                            'pw': float(f[5]), 'pw_max': float(f[6]), 'temp': float(f[7])})
        except Exception:
            pass
        _GPU_CACHE.update(t=time.time(), data=out)
        return out


def live_runs() -> dict:
    """Tags with a training process attached, mapped to what its command line reveals.

    Returning the arguments as well as the name matters: a run that started before metadata
    files existed, or whose log went somewhere we cannot see, still tells us its step budget
    right there in its command line -- which is enough to show progress and a finish estimate.
    """
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             r"Where-Object { $_.CommandLine -match '(train_run|mlm_run)\.py' } | "
             "ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True, timeout=8).stdout
    except Exception:
        return {}

    found = {}
    for line in out.splitlines():
        def arg(name, cast=str):
            m = re.search(rf'--{name}\s+(\S+)', line)
            return cast(m.group(1)) if m else None

        # Defaults matter here: a command line that omits --batch or --seq-len is using
        # mlm_run's defaults, and treating "absent" as "unknown" left the finish estimate blank
        # for exactly the runs launched the simple way.
        info = {'steps': arg('steps', int), 'batch': arg('batch', int) or 128,
                'seq_len': arg('seq-len', int) or 128, 'preset': arg('preset') or 'poc',
                'corpus': arg('corpus')}
        tag = arg('tag')
        if not tag:
            tokens = arg('tokens', int)
            if not (info['corpus'] and tokens and info['steps']):
                continue
            try:
                import mlm_train
                tag = mlm_train.cell_tag(info['corpus'], tokens, info['steps'],
                                         arg('seed', int) or 0, info['preset'] or 'poc')
            except Exception:
                continue
        found[tag] = info
    return found


def read_curve(tag: str) -> list[dict]:
    rows = []
    try:
        with open(os.path.join(RUNS, f'{tag}.jsonl'), encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except (OSError, ValueError):
        pass
    return rows


def _log_field(tag: str, pattern: str, cast=str):
    try:
        with open(os.path.join(LOGS, f'{tag}.log'), errors='replace') as f:
            for line in f:
                m = re.search(pattern, line)
                if m:
                    return cast(m.group(1).replace(',', ''))
    except OSError:
        pass
    return None


def snapshot(hours: float) -> dict:
    """Everything the page draws, in one JSON payload."""
    alive = live_runs()
    cutoff = time.time() - hours * 3600
    tags = set(alive)
    for p in glob.glob(os.path.join(LOGS, '*.log')):
        try:
            if os.path.getmtime(p) > cutoff:
                tags.add(os.path.basename(p)[:-4])
        except OSError:
            pass

    runs = []
    for tag in sorted(tags):
        rows = read_curve(tag)
        if not rows:
            continue
        last = rows[-1]

        # Prefer the metadata file. Regexing the console log only worked for runs whose log
        # happened to be where we looked, which is why finished-time estimates kept coming back
        # empty. Fall back to the log for runs that predate the meta file.
        meta = {}
        mp = os.path.join(RUNS, f'{tag}_meta.json')
        if os.path.exists(mp):
            try:
                with open(mp, encoding='utf-8') as f:
                    meta = json.load(f)
            except (OSError, ValueError):
                pass
        cli = alive.get(tag) or {}
        step_total = (meta.get('steps') or cli.get('steps')
                      or _log_field(tag, r'/([\d,]+) steps', int))
        total_work = meta.get('total_work') or _log_field(tag, r'total work ([\d,]+) tokens', int)
        if not total_work and step_total and cli.get('batch') and cli.get('seq_len'):
            total_work = step_total * cli['batch'] * cli['seq_len']
        # Infer the corpus from the tag when nothing recorded it, and take the vocabulary from
        # its stats file. Otherwise every run older than the meta file shows no verdict at all,
        # which is most of what you are looking at on any given day.
        corpus = meta.get('corpus') or cli.get('corpus') or tag.split('_')[0]
        random_loss = meta.get('random_loss')
        if random_loss is None:
            sp = os.path.join(HERE, 'data', corpus, 'stats.json')
            try:
                with open(sp, encoding='utf-8') as f:
                    random_loss = math.log(json.load(f)['vocab_size'])
            except (OSError, ValueError, KeyError):
                random_loss = None

        frac = min(1.0, last['step'] / step_total) if (step_total and last.get('step')) else None

        tps = [r['train']['tok_s'] for r in rows if 'train' in r and r['train'].get('tok_s')]
        med_tps = sorted(tps)[len(tps) // 2] if tps else None

        # Wall-clock keeps moving between logged points. Showing only the last logged elapsed
        # made a healthy run look frozen for the four to seven minutes between them.
        try:
            since = time.time() - os.path.getmtime(os.path.join(RUNS, f'{tag}.jsonl'))
        except OSError:
            since = 0.0
        live = tag in alive
        elapsed = (last.get('elapsed') or 0) + (since if live else 0)

        eta = None
        if live and total_work and med_tps and frac is not None and frac < 1:
            eta = total_work * (1 - frac) / med_tps

        # A plain-language read, so the page says what it thinks rather than only what it saw.
        state = 'stopped'
        if live:
            recent = [r['val']['loss'] for r in rows[-4:]]
            moved_total = rows[0]['val']['loss'] - last['val']['loss']
            moved_recent = (recent[0] - recent[-1]) if len(recent) > 1 else 0.0
            near_random = random_loss and last['val']['loss'] > random_loss - 3.0
            if moved_total < 0.15 and len(rows) > 4:
                state = 'stalled'
            elif moved_recent < 0.005 and len(rows) > 4:
                state = 'converged'
            elif near_random:
                state = 'on the plateau'
            else:
                state = 'learning'
        elif frac is not None and frac >= 0.999:
            state = 'done'

        result = None
        rp = os.path.join(RUNS, f'{tag}_result.json')
        if os.path.exists(rp):
            try:
                with open(rp, encoding='utf-8') as f:
                    r = json.load(f)
                result = {k: r.get(k) for k in ('val_loss', 'seconds', 'tokens_per_s',
                                                'stalled', 'lr_used', 'n_tokens',
                                                'preset', 'batch', 'steps')}
            except (OSError, ValueError):
                pass

        # The finished record is the most reliable source of the step count for older runs:
        # meta did not exist yet, the process is gone, and the console header writes
        # "11,719 steps x ..." which the log pattern (expecting "/N steps") never matched.
        if not step_total and result and result.get('steps'):
            step_total = result['steps']
            if frac is None:
                frac = 1.0

        runs.append({
            'tag': tag, 'live': live, 'frac': frac, 'state': state,
            'step': last.get('step'), 'steps': step_total,
            'train_loss': last['train']['loss'], 'val_loss': last['val']['loss'],
            'random_loss': random_loss, 'lr': last.get('lr'),
            'elapsed': elapsed, 'since_point': since, 'log_every': meta.get('log_every'),
            'tok_s': med_tps, 'eta_s': eta,
            'corpus': corpus,
            'steps_total': step_total,
            'study': bool(STUDY_TAG.match(tag)),
            'n_tokens': meta.get('n_tokens') or (result or {}).get('n_tokens'),
            'seed': meta.get('seed'),
            'accum': meta.get('accum'),
            'preset': (meta.get('preset') or cli.get('preset')
                       or (result or {}).get('preset')),
            'batch': (meta.get('batch') or cli.get('batch')
                      or (result or {}).get('batch')),
            'stalled': (result or {}).get('stalled'),
            'curve': [{'x': r.get('step') or r['epoch'],
                       'train': r['train']['loss'], 'val': r['val']['loss']} for r in rows],
            'result': result,
        })

    # "Is this good?" needs an anchor at each end. Zero is a model that learned nothing
    # (loss = log vocabulary); one hundred is the best any run on this corpus has reached. A
    # bare loss of 5.4 tells a reader nothing; "12% of the way to our best" tells them plenty.
    best_by_corpus = {}
    for r in runs:
        c = r.get('corpus')
        if c and r['val_loss'] is not None:
            best_by_corpus[c] = min(best_by_corpus.get(c, 9e9), r['val_loss'])
    for r in runs:
        rl, best = r.get('random_loss'), best_by_corpus.get(r.get('corpus'))
        if rl and best is not None and rl > best:
            r['quality'] = max(0.0, min(1.0, (rl - r['val_loss']) / (rl - best)))
            r['best_seen'] = best
        else:
            r['quality'] = None
            r['best_seen'] = None

    runs.sort(key=lambda r: (not r['live'], r['tag']))
    return {'now': time.time(), 'server_started': SERVER_STARTED,
            'gpus': gpus(), 'runs': runs}


class Handler(BaseHTTPRequestHandler):
    hours = 18.0

    def log_message(self, *a):
        pass                      # a dashboard should not spam the console it runs in

    def do_GET(self):
        if self.path.startswith('/api/'):
            body = json.dumps(snapshot(self.hours)).encode()
            ctype = 'application/json'
        else:
            body = PAGE.encode()
            ctype = 'text/html; charset=utf-8'
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>training</title>
<style>
:root{
  --bg:#0e1116; --card:#161b22; --line:#232a34; --ink:#e6edf3; --dim:#8b949e;
  --blue:#3987e5; --orange:#d95926; --aqua:#199e70; --red:#e66767; --amber:#d9a441;
}
@media (prefers-color-scheme:light){:root{
  --bg:#fbfcfd; --card:#fff; --line:#e3e8ee; --ink:#10151c; --dim:#6b7684;
  --blue:#2a78d6; --orange:#eb6834; --aqua:#1baf7a; --red:#d13438; --amber:#a06800;
}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{display:flex;align-items:baseline;gap:1rem;flex-wrap:wrap;
  padding:1rem 1.25rem;border-bottom:1px solid var(--line)}
h1{font-size:1rem;font-weight:650;margin:0;letter-spacing:.02em}
.muted{color:var(--dim);font-size:.82rem}
.help{padding:.7rem 1.25rem;border-bottom:1px solid var(--line);color:var(--dim);
  font-size:.8rem;line-height:1.6;max-width:80ch}
.help b{color:var(--ink);font-weight:600}
main{padding:1.25rem;display:flex;flex-direction:column;gap:1rem;max-width:1400px;margin:0 auto}
.gpus{display:grid;gap:.75rem;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
.gpu{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.75rem .9rem}
.gpu .top{display:flex;justify-content:space-between;align-items:baseline;gap:.5rem}
.gpu .nm{font-weight:600;font-size:.82rem}
.meter{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:.5rem 0 .35rem}
.meter>i{display:block;height:100%;background:var(--blue);border-radius:3px;
  transition:width .4s ease}
/* progress we cannot compute: a moving stripe, so it never reads as "finished" */
.meter.unknown>i{background:repeating-linear-gradient(90deg,var(--blue) 0 8px,
  transparent 8px 16px);opacity:.5;animation:slide 1.2s linear infinite}
@keyframes slide{to{background-position:16px 0}}
@media (prefers-reduced-motion:reduce){.meter.unknown>i{animation:none}}
.stats{display:flex;gap:.9rem;font-size:.76rem;color:var(--dim);font-variant-numeric:tabular-nums}
.runs{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(380px,1fr))}
.run svg{max-height:190px}
.run{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.9rem 1rem}
.run.livewire{border-color:color-mix(in oklab,var(--blue) 55%,var(--line))}
.run.stall{border-color:color-mix(in oklab,var(--amber) 60%,var(--line))}
.rhead{display:flex;justify-content:space-between;align-items:baseline;gap:.6rem;flex-wrap:wrap}
.tag{font-weight:650;font-size:.9rem;word-break:break-all}
.pill{font-size:.68rem;letter-spacing:.06em;text-transform:uppercase;
  padding:.15rem .45rem;border-radius:20px;border:1px solid var(--line);white-space:nowrap}
.pill.on{color:var(--aqua);border-color:color-mix(in oklab,var(--aqua) 50%,var(--line))}
.pill.off{color:var(--dim)}
.pill.warn{color:var(--amber);border-color:color-mix(in oklab,var(--amber) 50%,var(--line))}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(88px,1fr));gap:.5rem .75rem;
  margin:.7rem 0 .5rem;font-variant-numeric:tabular-nums}
.k{font-size:.66rem;color:var(--dim);letter-spacing:.04em;text-transform:uppercase}
.sub{font-size:.68rem;color:var(--dim);font-weight:400;margin-top:.1rem}
.v{font-size:1rem;font-weight:600}
.v.good{color:var(--aqua)} .v.mid{color:var(--amber)} .v.bad{color:var(--red)}
svg{display:block;width:100%;height:auto;overflow:visible}
.axis{stroke:var(--line);stroke-width:1}
.gl{stroke:var(--line);stroke-width:1;stroke-dasharray:2 3}
.tk{fill:var(--dim);font-size:9px;font-variant-numeric:tabular-nums}
.lg{display:flex;gap:.9rem;font-size:.72rem;color:var(--dim);margin-top:.35rem}
.sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:.3rem;
  vertical-align:baseline}
.empty{color:var(--dim);padding:2rem;text-align:center}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:1rem 1.1rem}
.chd{display:flex;align-items:baseline;gap:.8rem;flex-wrap:wrap}
.chd h2{font-size:.95rem;font-weight:650;margin:0}
.ctitle{font-size:.76rem;color:var(--dim);margin:.6rem 0 .1rem}
.cmpgrid{display:grid;gap:1rem;grid-template-columns:1fr}
@media (min-width:900px){.cmpgrid{grid-template-columns:minmax(0,4fr) minmax(0,6fr)}}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.6rem .9rem}
summary{cursor:pointer;font-size:.82rem;color:var(--dim);list-style:none}
summary::-webkit-details-marker{display:none}
summary::before{content:"▸ ";color:var(--dim)}
details[open] summary::before{content:"▾ "}
table.past{width:100%;border-collapse:collapse;margin-top:.6rem;font-size:.8rem;
  font-variant-numeric:tabular-nums}
table.past td{padding:.3rem .5rem;border-top:1px solid var(--line)}
table.past td:first-child{word-break:break-all}
table.past tr.hd td{color:var(--dim);font-size:.68rem;text-transform:uppercase;
  letter-spacing:.04em;border-top:none}
table.past td.good{color:var(--aqua)} table.past td.mid{color:var(--amber)}
table.past td.bad{color:var(--red)}
table.past td.num{text-align:right;color:var(--dim)}
</style></head><body>
<header>
  <h1>training</h1>
  <span class="muted" id="sub">connecting…</span>
  <span class="muted" style="margin-left:auto" id="clock"></span>
</header>
<div class="help">
  <b>Reading this:</b> the blue line is loss on held-out text (lower is better), orange is
  training loss. The dashed line is what a model that learned nothing scores — a curve still
  near it has not started learning yet. <b>Choosing between</b> turns the loss into the number
  of words the model is still undecided over, out of the whole vocabulary.
  <b>How good</b> places a run between those two anchors: 0% is a model that learned nothing,
  100% is the best any run has reached on the same corpus.
</div>
<main>
  <section class="gpus" id="gpus"></section>
  <section class="compare card" id="compare" hidden>
    <div class="chd"><h2>All runs together</h2><span class="muted" id="cmpnote"></span></div>
    <div class="cmpgrid">
      <div><div class="ctitle" id="lt"></div><div id="bars"></div></div>
      <div><div class="ctitle" id="rt"></div><div id="curves"></div></div>
    </div>
    <div class="lg" id="cmpleg"></div>
  </section>
  <section class="runs" id="runs"></section>
  <div class="empty" id="empty" hidden>No runs in the window. Start one and it appears here.</div>
  <details id="past" hidden>
    <summary><span id="pastn"></span></summary>
    <table class="past"><tbody id="pastrows"></tbody></table>
  </details>
</main>
<script>
// 0% = learned nothing, 100% = as good as the best run we have on this corpus.
const qcls = q => q==null ? '' : q>=0.85 ? 'good' : q>=0.35 ? 'mid' : 'bad';
const fmtT = s => s==null ? "–" : s<60 ? Math.round(s)+"s"
  : s<3600 ? Math.floor(s/60)+"m"+String(Math.round(s%60)).padStart(2,"0")+"s"
  : Math.floor(s/3600)+"h"+String(Math.floor((s%3600)/60)).padStart(2,"0")+"m";
const fmtN = n => n==null ? "–" : n>=1e6 ? (n/1e6).toFixed(1)+"M"
  : n>=1e3 ? Math.round(n/1e3)+"k" : String(Math.round(n));

// One curve panel: train and val loss against step. Two series, so a legend is always present;
// the last point of each is marked so the current value is findable without a tooltip.
function chart(curve, randomLoss, stepsTotal){
  if(!curve || curve.length<2) return '<div class="muted" style="padding:.5rem 0">no points yet</div>';
  const W=520,H=132,L=42,R=10,T=10,B=22;
  const xs=curve.map(p=>p.x), ys=curve.flatMap(p=>[p.train,p.val]);
  if(randomLoss && randomLoss < Math.max(...ys)+2) ys.push(randomLoss);
  // Span the full step budget when we know it. Auto-scaling to the data drew a
  // 40%-complete run edge to edge, which reads as finished.
  const x0=0, x1=stepsTotal || Math.max(...xs);
  let y0=Math.min(...ys), y1=Math.max(...ys);
  const pad=(y1-y0)*0.12||0.1; y0-=pad; y1+=pad;
  const X=v=>L+(v-x0)/((x1-x0)||1)*(W-L-R);
  const Y=v=>T+(y1-v)/((y1-y0)||1)*(H-T-B);
  const path=k=>curve.map((p,i)=>(i?"L":"M")+X(p.x).toFixed(1)+" "+Y(p[k]).toFixed(1)).join(" ");
  let g="";
  for(let i=0;i<=3;i++){
    const v=y0+(y1-y0)*i/3, y=Y(v).toFixed(1);
    g+=`<line class="gl" x1="${L}" y1="${y}" x2="${W-R}" y2="${y}"/>`
     + `<text class="tk" x="${L-6}" y="${(+y+3).toFixed(1)}" text-anchor="end">${v.toFixed(2)}</text>`;
  }
  // The loss a model that learned nothing would score. Without it a reader has no idea
  // whether 5.4 is good, and every curve looks like it is descending nicely.
  let ref="";
  if(randomLoss && randomLoss<=y1 && randomLoss>=y0){
    const ry=Y(randomLoss).toFixed(1);
    ref=`<line x1="${L}" y1="${ry}" x2="${W-R}" y2="${ry}" stroke="var(--dim)"
          stroke-width="1" stroke-dasharray="4 4" opacity=".7"/>
      <text class="tk" x="${W-R-2}" y="${(+ry-4).toFixed(1)}" text-anchor="end"
        >learned nothing</text>`;
  }
  const lastT=curve[curve.length-1], marks=
      `<circle cx="${X(lastT.x)}" cy="${Y(lastT.train)}" r="3" fill="var(--orange)"/>`
    + `<circle cx="${X(lastT.x)}" cy="${Y(lastT.val)}" r="3" fill="var(--blue)"/>`;
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="loss curve">
    ${g}${ref}<line class="axis" x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}"/>
    <path d="${path('train')}" fill="none" stroke="var(--orange)" stroke-width="1.8"
          stroke-linejoin="round"/>
    <path d="${path('val')}" fill="none" stroke="var(--blue)" stroke-width="2.2"
          stroke-linejoin="round"/>${marks}
    <text class="tk" x="${L}" y="${H-6}">0</text>
    <text class="tk" x="${X(lastT.x)}" y="${H-6}" text-anchor="middle">${fmtN(lastT.x)}</text>
    <text class="tk" x="${W-R}" y="${H-6}" text-anchor="end">${fmtN(x1)} steps</text>
  </svg>
  <div class="lg"><span><i class="sw" style="background:var(--blue)"></i>val</span>
  <span><i class="sw" style="background:var(--orange)"></i>train (EMA)</span></div>`;
}

function gpuCard(g){
  const pct=Math.round(g.util);
  return `<div class="gpu"><div class="top"><span class="nm">cuda:${g.i}</span>
    <span class="muted">${pct}%</span></div>
    <div class="meter"><i style="width:${pct}%"></i></div>
    <div class="stats"><span>${g.mem.toFixed(1)}/${g.mem_tot.toFixed(0)} GB</span>
    <span>${Math.round(g.pw)}/${Math.round(g.pw_max)} W</span>
    <span>${Math.round(g.temp)}°C</span></div></div>`;
}

function runCard(r){
  const known = r.frac!=null;
  const pct = known ? Math.round(r.frac*100) : 0;
  const state = r.live ? '<span class="pill on">training</span>'
    : (r.frac>=0.999 ? '<span class="pill off">done</span>'
                     : '<span class="pill off">stopped</span>');
  const stall = r.stalled ? '<span class="pill warn">stalled</span>' : '';
  const box = 'run' + (r.live?' livewire':'')
    + ((r.stalled||r.state==='stalled'||r.state==='on the plateau')?' stall':'');
  return `<div class="${box}">
    <div class="rhead"><span class="tag">${r.tag}</span><span>${stall}${state}</span></div>
    <div class="meter ${known?'':'unknown'}"><i style="width:${known?pct:100}%"></i></div>
    <div class="grid">
      <div><div class="k">how good</div>
        <div class="v ${qcls(r.quality)}">${r.quality==null?'–':Math.round(r.quality*100)+'%'}</div>
        <div class="sub">${r.quality==null?'no reference yet':'of our best on '+(r.corpus||'this corpus')}</div></div>
      <div><div class="k">choosing between</div>
        <div class="v">${fmtN(Math.exp(r.val_loss))}</div>
        <div class="sub">words${r.random_loss?' of '+fmtN(Math.exp(r.random_loss)):''}</div></div>
      <div><div class="k">val loss</div><div class="v">${r.val_loss.toFixed(3)}</div>
        <div class="sub">train ${r.train_loss.toFixed(3)}</div></div>
      <div><div class="k">step</div>
        <div class="v">${fmtN(r.step)}${r.steps?'<span class="sub"> / '+fmtN(r.steps)+'</span>':''}</div></div>
      <div><div class="k">speed</div><div class="v">${fmtN(r.tok_s)}</div>
        <div class="sub">tokens/sec</div></div>
      <div><div class="k">elapsed</div><div class="v">${fmtT(r.elapsed)}</div>
        <div class="sub">${r.live?'updated '+fmtT(r.since_point)+' ago':'&nbsp;'}</div></div>
      <div><div class="k">remaining</div><div class="v">${r.live?fmtT(r.eta_s):'–'}</div>
        <div class="sub">${r.batch?'batch '+r.batch:'&nbsp;'}</div></div>
    </div>
    ${chart(r.curve, r.random_loss, r.steps_total)}</div>`;
}


// ---- the unified view --------------------------------------------------------------------
// Structured after the Part 1 chart, which was legible for reasons worth copying: the x axis is
// an ORDERED variable (there, images per class; here, unique tokens), bars are GROUPED so the
// two model sizes read as a pair rather than as neighbours, and every line on the right is named
// in a legend. Ad-hoc runs are excluded -- a probe sitting between two grid cells is what made
// the first attempt unreadable.
const HUES = ["#3987e5","#199e70","#9085e9","#d9a441","#d95926","#e66767"];
const SIZES = {poc: "33.8M", afriberta: "86M"};
const isBig = r => (r.preset || 'poc') !== 'poc';
const fmtTok = n => n >= 1e6 ? (n/1e6).toFixed(0)+'M' : fmtN(n);

// One row per (data rung x model size), seeds averaged, so three seeds of one cell are one bar
// rather than three bars pretending to be different configurations.
function cells(runs){
  const grid = {};
  for(const r of runs){
    if(!r.study || !r.n_tokens || r.val_loss == null) continue;
    // Key on the COMPUTE budget too. Without it, two runs at the same data rung and model size
    // but different step counts get averaged together and reported as "2 seeds" -- which is a
    // different experiment described as a repeat of the same one.
    const key = [r.n_tokens, r.preset || 'poc', r.steps_total || 0].join('|');
    (grid[key] = grid[key] || {n: r.n_tokens, preset: r.preset || 'poc',
                               steps: r.steps_total || 0, runs: []}).runs.push(r);
  }
  return Object.values(grid).map(c => {
    const v = c.runs.map(r => r.val_loss);
    c.val = v.reduce((a,b)=>a+b,0)/v.length;
    c.spread = v.length > 1 ? Math.max(...v) - Math.min(...v) : 0;
    c.seeds = v.length;
    c.live = c.runs.some(r => r.live);
    c.curve = c.runs.slice().sort((a,b)=>b.curve.length-a.curve.length)[0].curve;
    c.random = c.runs.map(r=>r.random_loss).find(Boolean);
    return c;
  }).sort((a,b) => a.n - b.n || (a.preset > b.preset ? 1 : -1) || a.steps - b.steps);
}

function rungColour(rungs){
  const m = {}; rungs.forEach((n,i)=> m[n] = HUES[i % HUES.length]); return m;
}

// LEFT: where each one lands. Grouped bars, x ordered by how much unique text the run saw.
function barsChart(cs, col, rungs){
  const W=460,H=280,L=52,R=10,T=18,B=66;
  if(!cs.length) return '<div class="muted">no grid runs yet</div>';
  const rl = cs.map(c=>c.random).find(Boolean);
  const maxV = Math.max(...cs.map(c=>c.val), rl||0)*1.06;
  const Y = v => T + (1 - v/maxV)*(H-T-B);
  const slot = (W-L-R)/rungs.length;

  let g='';
  for(let k=0;k<=4;k++){
    const v=maxV*k/4, y=Y(v).toFixed(1);
    g += `<line class="gl" x1="${L}" y1="${y}" x2="${W-R}" y2="${y}"/>`
       + `<text class="tk" x="${L-6}" y="${(+y+3).toFixed(1)}" text-anchor="end">${v.toFixed(1)}</text>`;
  }
  let bars='';
  rungs.forEach((n,gi)=>{
    const inRung = cs.filter(c=>c.n===n);
    const bw = Math.min(30, (slot-16)/Math.max(inRung.length,1));
    inRung.forEach((c,bi)=>{
      const x = L + gi*slot + (slot - bw*inRung.length)/2 + bi*bw;
      const y = Y(c.val);
      bars += `<rect x="${x}" y="${y}" width="${bw-3}" height="${Math.max(1,(H-B)-y)}" rx="2"
          fill="${col[n]}" opacity="${isBig(c)?0.45:1}"/>
        <text class="tk" x="${x+(bw-3)/2}" y="${y-4}" text-anchor="middle"
          style="font-weight:600">${c.val.toFixed(2)}</text>`;
    });
    bars += `<text class="tk" x="${L+gi*slot+slot/2}" y="${H-B+15}" text-anchor="middle"
        style="font-weight:600">${fmtTok(n)}</text>
      <text class="tk" x="${L+gi*slot+slot/2}" y="${H-B+27}" text-anchor="middle"
        >unique tokens</text>`;
  });
  const ref = rl ? `<line x1="${L}" y1="${Y(rl)}" x2="${W-R}" y2="${Y(rl)}" stroke="var(--dim)"
      stroke-dasharray="4 4" stroke-width="1"/>
      <text class="tk" x="${W-R}" y="${Y(rl)-4}" text-anchor="end">learned nothing (${rl.toFixed(1)})</text>` : '';
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="final loss by data rung">
    ${g}${ref}${bars}<line class="axis" x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}"/>
    <text class="tk" x="4" y="${T+4}">val loss</text></svg>`;
}

// RIGHT: how each one gets there. Same colours, dash = larger model, dot = stopped improving.
function curvesChart(cs, col){
  const W=560,H=280,L=46,R=12,T=18,B=40;
  const all = cs.filter(c=>c.curve && c.curve.length>1);
  if(!all.length) return '';
  const xs = all.flatMap(c=>c.curve.map(p=>p.x)).filter(x=>x>0);
  const ys = all.flatMap(c=>c.curve.map(p=>p.val));
  const rl = all.map(c=>c.random).find(Boolean);
  const x0=Math.max(1,Math.min(...xs)), x1=Math.max(...xs);
  let y0=Math.min(...ys), y1=Math.max(...ys.concat(rl?[rl]:[]));
  const pad=(y1-y0)*0.08||0.1; y0-=pad; y1+=pad;
  const lg = v => Math.log(Math.max(v,1));
  const X = v => L+(lg(v)-lg(x0))/((lg(x1)-lg(x0))||1)*(W-L-R);
  const Y = v => T+(y1-v)/((y1-y0)||1)*(H-T-B);

  let g='';
  for(let k=0;k<=4;k++){
    const v=y0+(y1-y0)*k/4, y=Y(v).toFixed(1);
    g += `<line class="gl" x1="${L}" y1="${y}" x2="${W-R}" y2="${y}"/>`
       + `<text class="tk" x="${L-6}" y="${(+y+3).toFixed(1)}" text-anchor="end">${v.toFixed(1)}</text>`;
  }
  const ticks=[];
  for(let e=0;e<8;e++) [1,3].forEach(m=>{const v=m*Math.pow(10,e); if(v>=x0&&v<=x1) ticks.push(v);});
  const xax = ticks.map(v=>`<text class="tk" x="${X(v)}" y="${H-B+14}"
      text-anchor="middle">${fmtN(v)}</text>`).join('');

  let lines='';
  for(const c of all){
    const pts=c.curve.filter(p=>p.x>0);
    if(pts.length<2) continue;
    const d=pts.map((p,i)=>(i?'L':'M')+X(p.x).toFixed(1)+' '+Y(p.val).toFixed(1)).join(' ');
    lines += `<path d="${d}" fill="none" stroke="${col[c.n]}" stroke-width="${c.live?2.8:2}"
        stroke-linejoin="round" stroke-dasharray="${isBig(c)?'5 4':'0'}"/>`;
    const f=flatIdx(pts);
    lines += `<circle cx="${X(pts[f].x)}" cy="${Y(pts[f].val)}" r="3.5" fill="${col[c.n]}"/>`;
  }
  const ref = rl ? `<line x1="${L}" y1="${Y(rl)}" x2="${W-R}" y2="${Y(rl)}" stroke="var(--dim)"
      stroke-dasharray="4 4" stroke-width="1"/>` : '';
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="training curves by configuration">
    ${g}${ref}${lines}<line class="axis" x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}"/>${xax}
    <text class="tk" x="${(L+W)/2}" y="${H-4}" text-anchor="middle">optimizer step (log scale)</text>
    <text class="tk" x="4" y="${T+4}">val loss</text></svg>`;
}

// The point after which a run stopped meaningfully improving.
function flatIdx(c){
  for(let i=1;i<c.length;i++){
    const rest=c.slice(i);
    if(rest.length<3) break;
    if((rest[0].val - rest[rest.length-1].val) < 0.02) return i;
  }
  return c.length-1;
}

function renderCompare(runs){
  const el=document.getElementById('compare');
  const cs=cells(runs);
  el.hidden = cs.length<2;
  if(el.hidden) return;
  const rungs=[...new Set(cs.map(c=>c.n))].sort((a,b)=>a-b);
  const col=rungColour(rungs);

  document.getElementById('bars').innerHTML   = barsChart(cs, col, rungs);
  document.getElementById('curves').innerHTML = curvesChart(cs, col);
  document.getElementById('lt').textContent =
    'Where each one lands  (left bar = 33.8M, right bar = 86M)';
  document.getElementById('rt').textContent =
    'How each one gets there  (dashed = 86M, dot = stopped improving)';

  const seeded = cs.filter(c=>c.seeds>1).length;
  document.getElementById('cmpnote').textContent =
    cs.length + ' configurations' + (seeded ? ', seeds averaged' : '');

  // Name every line, the way the Part 1 legend did. A colour with no name is a decoration.
  const multi = {};
  cs.forEach(c => { const k = c.n + '|' + c.preset; multi[k] = (multi[k]||0) + 1; });
  document.getElementById('cmpleg').innerHTML = cs.map(c =>
    '<span><i class="sw" style="background:' + col[c.n]
    + (isBig(c) ? ';opacity:.45' : '') + '"></i>'
    + fmtTok(c.n) + ' &middot; ' + (SIZES[c.preset] || c.preset)
    + (multi[c.n + '|' + c.preset] > 1 ? ' &middot; ' + fmtN(c.steps) + ' steps' : '')
    + (c.seeds > 1 ? ' (' + c.seeds + ' seeds)' : '') + '</span>').join('');
}

async function tick(){
  try{
    const d = await (await fetch('/api/', {cache:'no-store'})).json();
    document.getElementById('gpus').innerHTML = d.gpus.map(gpuCard).join('');
    // Runs still going get a full card. Everything finished collapses into one closed list --
    // on a busy day that is fifteen dead panels between you and the run you care about.
    renderCompare(d.runs);
    const live = d.runs.filter(r=>r.live), past = d.runs.filter(r=>!r.live);
    document.getElementById('runs').innerHTML = live.map(runCard).join('');
    document.getElementById('empty').hidden = d.runs.length>0;

    const pastEl = document.getElementById('past');
    pastEl.hidden = past.length===0;
    document.getElementById('pastn').textContent =
      `Earlier runs (${past.length}) — click to compare`;
    document.getElementById('pastrows').innerHTML =
      '<tr class="hd"><td>run</td><td class="num">how good</td><td class="num">val loss</td>'
      + '<td class="num">choosing between</td><td class="num">steps</td>'
      + '<td class="num">batch</td><td class="num">took</td></tr>'
      + past.map(r=>`<tr>
      <td>${r.tag}${r.stalled?' <span class="pill warn">stalled</span>':''}</td>
      <td class="num ${qcls(r.quality)}">${r.quality==null?'–':Math.round(r.quality*100)+'%'}</td>
      <td class="num">${r.val_loss.toFixed(3)}</td>
      <td class="num">${fmtN(Math.exp(r.val_loss))} words</td>
      <td class="num">${fmtN(r.steps_total)}</td>
      <td class="num">${r.batch||'–'}</td>
      <td class="num">${fmtT(r.elapsed)}</td></tr>`).join('');

    document.getElementById('sub').textContent = live.length
      ? `${live.length} training` : 'nothing training right now';
    // Show when the SERVER started. An edited file does not reach a process that is already
    // running, and a dashboard that silently serves stale code wastes a lot of confusion.
    const up = new Date(d.server_started*1000).toLocaleTimeString();
    document.getElementById('clock').textContent =
      `${new Date().toLocaleTimeString()} · server up since ${up}`;
  }catch(e){ document.getElementById('sub').textContent = 'lost connection to the server'; }
}
tick(); setInterval(tick, 3000);
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description='Browser dashboard for training runs.')
    p.add_argument('--port', type=int, default=8770)
    p.add_argument('--host', default='127.0.0.1',
                   help='0.0.0.0 to reach it from other machines (no auth -- trusted networks)')
    p.add_argument('--hours', type=float, default=18.0,
                   help='how far back to include finished runs')
    a = p.parse_args()

    Handler.hours = a.hours
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    where = 'localhost' if a.host in ('127.0.0.1', 'localhost') else a.host
    print(f'  dashboard  http://{where}:{a.port}')
    if a.host == '0.0.0.0':
        try:
            import socket
            print(f'  on this network  http://{socket.gethostbyname(socket.gethostname())}:'
                  f'{a.port}')
        except Exception:
            pass
    print('  read-only; Ctrl+C to stop\n')
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('stopped')


if __name__ == '__main__':
    main()
