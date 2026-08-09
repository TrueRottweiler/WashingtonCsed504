"""
dashboard.py, one live screen for the whole A2 language-model fleet.

Duplicated from a1-cv/dashboard.py; same read-only design -- we never touch the training
processes, we only read runs/*.jsonl plus the live tqdm tail of logs/*.log and fold in a quick
nvidia-smi. The adaptations are the scoreboard's: the headline metric is validation perplexity,
which improves DOWNWARD, so "best" is a minimum, the sparkline falls as the model learns, and
the overfitting grade is the val/train perplexity ratio instead of an accuracy gap.

Usage:
    python dashboard.py                 # redraw every 2s until you hit Ctrl+C
    python dashboard.py --once          # print one snapshot and exit
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Force our stdout to UTF-8 (a cp1252 Windows console cannot encode a sparkline); fall back to
# ASCII glyphs on a genuinely legacy console rather than crashing mid-frame.
UNICODE = True
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    UNICODE = False

HERE = os.path.dirname(os.path.abspath(__file__))

# Set from the command line in main(). Watching a single run should not mean scrolling past
# fifteen finished ones.
ONLY_LIVE = False
TAG_FILTER = None
RECENT_HOURS = 18.0
RUNS, LOGS = os.path.join(HERE, 'runs'), os.path.join(HERE, 'logs')

# Same scheme as the a1 dashboard and the notebooks: one hue per dataset, the recurrent model in
# the darker shade and the transformer in the lighter one -- the pair worth comparing shares a
# hue, the datasets (not comparable to each other) are told apart by hue instead.
PALETTE = {
    'shakespeare': ('dark_orange3',   'orange1'),
    'wikitext2':   ('deep_sky_blue4', 'sky_blue2'),
    'wikitext103': ('green4',         'aquamarine3'),
}


def color_for(tag):
    """The run's color: dataset picks the hue, family picks the shade (lstm dark, gpt light)."""
    for dataset, (dark, light) in PALETTE.items():
        if tag.startswith(dataset + '_'):
            model = tag[len(dataset) + 1:]
            return light if model.startswith('gpt') else dark
    return 'white'


SPARK = ' ▁▂▃▄▅▆▇█' if UNICODE else ' .:-=+*#%'
FULL, EMPTY = ('█', '░') if UNICODE else ('#', '.')


def sparkline(vals, width=28) -> str:
    """The val-perplexity curve as a one-line sparkline. Perplexity falls as the model learns,
    so a healthy run reads high-to-low, the mirror image of the a1 accuracy sparkline."""
    if not vals:
        return ''
    v = vals[-width:]
    lo, hi = min(v), max(v)
    if hi - lo < 1e-9:
        return SPARK[1] * len(v)
    return ''.join(SPARK[min(8, int((x - lo) / (hi - lo) * 8) + 1)] for x in v)


def read_jsonl(tag):
    p = os.path.join(RUNS, f'{tag}.jsonl')
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a half-written final line; complete on the next tick
    return out


def read_progress(tag):
    """Scrape the current epoch's progress out of the live tqdm bar at the tail of the log."""
    p = os.path.join(LOGS, f'{tag}.log')
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 4000))
        tail = f.read().decode('utf-8', 'replace')
    frames = tail.replace('\r', '\n').split('\n')
    for line in reversed(frames):
        m = re.search(r'(\d+)/(\d+)\s*\[', line)
        if m:
            cur, tot = int(m.group(1)), int(m.group(2))
            tps = re.search(r'([\d.]+)k tok/s', line)
            return {'cur': cur, 'tot': tot, 'frac': cur / max(1, tot),
                    'tok_s': float(tps.group(1)) * 1000 if tps else None}
    return None


def gpus():
    try:
        q = ('index,utilization.gpu,memory.used,memory.total,power.draw,power.limit,'
             'temperature.gpu')
        out = subprocess.run(['nvidia-smi', f'--query-gpu={q}', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        rows = []
        for line in out.splitlines():
            i, u, mu, mt, pd, pl, t = [x.strip() for x in line.split(',')]
            rows.append({'i': int(i), 'util': float(u), 'mem': float(mu) / 1024,
                         'mem_tot': float(mt) / 1024, 'pw': float(pd), 'pw_max': float(pl),
                         'temp': float(t)})
        return rows
    except Exception:
        return []


def live_tags():
    """Which A2 runs have a training process actually running right now?

    Matches BOTH runners. The causal study uses train_run.py and the masked-LM study uses
    mlm_run.py; this only looked for the first, so every MLM run was reported STOPPED while it
    was in fact training, and the whole display filled with red.

    Same tag reconstruction as a1's dashboard, with one addition: we must not claim a1-cv's
    runs as ours (both assignments have a train_run.py). A fleet-launched child has the full
    a2-nlp path in its command line, but a run started by hand from inside this folder says
    just 'train_run.py' -- the path test alone once drew two live, resumed runs as STOPPED.
    So we accept either signal: the a2-nlp path, or an --dataset that belongs to this
    assignment (the two assignments' dataset names do not overlap).
    """
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             r"Where-Object { $_.CommandLine -match '(train_run|mlm_run)\.py' } | "
             "ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True, timeout=8).stdout
        tags = set()
        for line in out.splitlines():
            # mlm_run.py says --corpus where train_run.py says --dataset.
            md = re.search(r'--(?:dataset|corpus)\s+(\S+)', line)
            ours = 'a2-nlp' in line or (md and md.group(1) in PALETTE)
            if not ours:
                continue
            mt = re.search(r'--tag\s+(\S+)', line)
            if mt:
                base = mt.group(1)
            else:
                mm = re.search(r'--model\s+(\S+)', line)
                if mm:
                    model = mm.group(1)
                    md2 = re.search(r'--dataset\s+(\S+)', line)
                    base = f'{(md2.group(1) if md2 else "wikitext2")}_{model}'
                else:
                    # An MLM run names itself from corpus/tokens/steps/seed/preset rather than
                    # from a model name, and the fleet does not pass --tag. Rebuild it with the
                    # same function the runner uses, so the two cannot disagree.
                    mc = re.search(r'--corpus\s+(\S+)', line)
                    mtok = re.search(r'--tokens\s+(\d+)', line)
                    mst = re.search(r'--steps\s+(\d+)', line)
                    if not (mc and mtok and mst):
                        continue
                    msd = re.search(r'--seed\s+(\d+)', line)
                    mpr = re.search(r'--preset\s+(\S+)', line)
                    try:
                        import mlm_train
                        base = mlm_train.cell_tag(
                            mc.group(1), int(mtok.group(1)), int(mst.group(1)),
                            int(msd.group(1)) if msd else 0,
                            mpr.group(1) if mpr else 'poc')
                    except Exception:
                        continue
            smoke = '--smoke-test' in line or '--smoke' in line
            tags.add(f'smoke-{base}' if smoke else base)
        return tags
    except Exception:
        return set()


def bar(frac, width=18, color='white'):
    n = int(max(0.0, min(1.0, frac)) * width)
    return Text(FULL * n + EMPTY * (width - n), style=color)


_REF_TPS_CACHE = {}
_N_TOKENS_CACHE = {}


def _n_train_tokens(dataset):
    """Train-split token count from the dataset's own stats.json, cached; None if not prepared.

    Returns None for an unknown dataset rather than raising. _run_meta yields None whenever a
    log has no recognizable dataset header, and the whole dashboard is a read-only view of other
    processes' output -- it should degrade to "no prediction" for a run it cannot parse, never
    take the display down. It previously died with a TypeError from os.path.join(None).
    """
    if not dataset:
        return None
    if dataset not in _N_TOKENS_CACHE:
        try:
            with open(os.path.join(HERE, 'data', dataset, 'stats.json')) as f:
                _N_TOKENS_CACHE[dataset] = json.load(f)['n_tokens']['train']
        except (OSError, KeyError, ValueError, TypeError):
            _N_TOKENS_CACHE[dataset] = None
    return _N_TOKENS_CACHE[dataset]


def _ref_tps(model):
    """This model's characteristic throughput (median tok/s) from a prior completed run -- the
    'previous stats' the Predicted ETA is built on. Throughput transfers across datasets for the
    same reason as a1: the per-token work depends on the model, not on which corpus the tokens
    came from."""
    if model not in _REF_TPS_CACHE:
        tps = None
        for path in glob.glob(os.path.join(RUNS, f'*_{model}_result.json')) + \
                glob.glob(os.path.join(RUNS, f'*_{model}_s*_result.json')):
            try:
                d = json.load(open(path))
                vals = sorted(e['train']['tok_s'] for e in d.get('history', [])
                              if 'train' in e and 'tok_s' in e['train'])
                if vals:
                    tps = vals[len(vals) // 2]
                    break
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        _REF_TPS_CACHE[model] = tps
    return _REF_TPS_CACHE[model]


def _declared_work(tag):
    """Total tokens a run says it will process, from its own header. None if it does not say."""
    try:
        with open(os.path.join(LOGS, f'{tag}.log'), errors='replace') as f:
            for line in f:
                m = re.search(r'total work ([\d,]+) tokens', line)
                if m:
                    return int(m.group(1).replace(',', ''))
    except OSError:
        pass
    return None


def _observed_tps(tag):
    """This run's own median throughput so far -- better than a prior run's for its own ETA."""
    rows = read_jsonl(tag)
    vals = sorted(r['train']['tok_s'] for r in rows
                  if 'train' in r and r['train'].get('tok_s'))
    return vals[len(vals) // 2] if vals else None


def _predict_total_s(dataset, model, total_epochs, tag=None):
    """Predicted wall-clock for the whole run from a prior run's throughput -- a genuine
    prediction to hold the live estimate against, not a restatement of it."""
    # A run that declares its own total work gets an exact answer: divide by the throughput it
    # is actually achieving. Only fall back to the epochs x corpus-size heuristic, which assumes
    # one epoch covers the corpus once, when the run says nothing.
    work = _declared_work(tag) if tag else None
    if work:
        tps = _observed_tps(tag) or _ref_tps(model)
        if tps:
            return work / tps + 15.0

    ref = _ref_tps(model)
    n_train = _n_train_tokens(dataset)
    if not ref or not n_train:
        return None
    return total_epochs * n_train / ref + 15.0


def render(t0):
    """Build the frame: a hardware panel on top, then one stacked card per run."""
    alive = live_tags()
    recent = time.time() - RECENT_HOURS * 3600
    tags = set(alive)
    if not ONLY_LIVE:
        for p in glob.glob(os.path.join(LOGS, '*.log')):
            try:
                if os.path.getmtime(p) > recent:
                    tags.add(os.path.basename(p)[:-4])
            except OSError:
                pass
    if TAG_FILTER:
        tags = {t for t in tags if TAG_FILTER in t}
    tags = sorted(tags)

    blocks = []

    g = Table.grid(padding=(0, 2))
    for d in gpus():
        us = 'bold green' if d['util'] > 85 else ('yellow' if d['util'] > 40 else 'red')
        hot = 'bold red' if d['temp'] >= 90 else ('yellow' if d['temp'] >= 87 else 'green')
        g.add_row(Text(f"cuda:{d['i']}", style='bold cyan'),
                  bar(d['util'] / 100, 12, us) + Text(f" {d['util']:3.0f}%", style=us),
                  Text(f"{d['mem']:5.1f}/{d['mem_tot']:.0f}GB"),
                  Text(f"{d['pw']:3.0f}/{d['pw_max']:.0f}W"),
                  Text(f"{d['temp']:.0f}C", style=hot))
    blocks.append(Panel(g, title='[bold]hardware', border_style='gray37', padding=(0, 1)))

    for tag in tags:
        rows = read_jsonl(tag)
        prog = read_progress(tag)
        col = color_for(tag)
        running = tag in alive
        total = _total_epochs(tag)
        gpu, params, dataset = _run_meta(tag)

        title = f'[bold {col}]{tag}[/]'
        meta = (([dataset] if dataset else [])
                + ([f'{params/1e6:.0f}M param'] if params else [])
                + ([f'cuda:{gpu}'] if gpu is not None else []))
        if meta:
            title += '   [dim]' + '  '.join(meta) + '[/]'

        if not rows:
            blocks.append(Panel(Text('starting...', style='yellow'),
                                title=title, border_style='gray37', padding=(0, 1)))
            continue

        # The latest finished epoch. Perplexity improves downward, so best is the minimum and
        # every comparison below is the mirror image of the a1 dashboard's.
        last = rows[-1]
        ppls = [r['val']['ppl'] for r in rows]
        best = min(ppls)
        best_ep = ppls.index(best) + 1
        ep = last['epoch']
        tr_ppl, va_ppl = last['train']['ppl'], last['val']['ppl']

        # Headline state. REGRESSING here means val perplexity is RISING while the run is alive
        # -- past the best epoch that is ordinary late-run overfit, but worth an eyeball.
        if not running:
            state_txt, state_sty = ('DONE', 'bold green') if ep >= total else ('STOPPED', 'bold red')
        elif len(ppls) >= 3 and ppls[-1] > ppls[-3] * 1.02:
            state_txt, state_sty = 'REGRESSING', 'bold yellow'
        else:
            state_txt, state_sty = 'training', 'green'

        # The overfitting grade: the val/train perplexity ratio. Both numbers are the same clean
        # next-token exam here (no mixup asterisk like a1), so the ratio is honest: near 1 is
        # healthy, and a val perplexity several times train means the model is memorizing the
        # training stream. wikitext2 runs WILL go red late -- that is the small rung being small.
        ratio = va_ppl / max(1e-9, tr_ppl)
        if ratio > 3.0:
            health_txt, health_sty = f'val/train {ratio:4.1f}x MEMORIZING', 'bold red'
        elif ratio > 1.8:
            health_txt, health_sty = f'val/train {ratio:4.1f}x overfitting', 'yellow'
        else:
            health_txt, health_sty = f'val/train {ratio:4.1f}x healthy', 'green'

        frac = (ep + (prog['frac'] if prog else 0)) / max(1, total)
        tps = (prog or {}).get('tok_s') or last['train']['tok_s']
        elapsed = last['elapsed']
        k = min(5, len(rows) - 1)
        per_ep = (elapsed - rows[-k - 1]['elapsed']) / k if k > 0 else elapsed
        remaining = _fmt((total - ep) * per_ep) if (running and ep < total) else '-'
        model = tag[len(dataset) + 1:] if (dataset and tag.startswith(dataset + '_')) else tag
        model = re.sub(r'_s\d+$', '', model)
        pred_s = _predict_total_s(dataset, model, total, tag)
        predicted = _fmt(pred_s) if pred_s else '?'
        lr = last.get('lr')

        L1 = bar(frac, 30, col) + Text.assemble(
            '  ', (f'ep {ep:>3}/{total:<4}', 'bold'), '   ',
            (f'{state_txt:<11}', state_sty), (f'{tps/1000:>7.0f}k tok/s', 'dim'))
        L2 = Text.assemble(
            ('val   ', 'dim'), (f'ppl {va_ppl:9.2f}', col),
            ('    best ', 'dim'), (f'{best:9.2f} @ep{best_ep:<4}', 'bold'),
            (health_txt, health_sty))
        L3 = Text.assemble(
            ('train ', 'dim'), (f'ppl {tr_ppl:9.2f}', 'dim'),
            ('    lr ', 'dim'), (f'{lr:8.5f}' if lr is not None else '   -    ', 'dim'),
            ('    loss ', 'dim'), (f'{last["train"]["loss"]:6.3f}', 'dim'))
        L4 = Text.assemble(
            ('time  ', 'dim'), ('elapsed ', 'dim'), (f'{_fmt(elapsed):<6}', ''),
            ('  remaining ', 'dim'), (f'{remaining:<6}', 'cyan'),
            ('  predicted ', 'dim'), (f'{predicted:<6}', 'magenta'),
            ('  ', 'dim'), (f'{per_ep:.1f}s/ep', 'dim'))
        L5 = Text(sparkline(ppls, 60), style=col)
        blocks.append(Panel(Group(L1, L2, L3, L4, L5), title=title,
                            border_style='gray37', padding=(0, 1)))

    blocks.append(Text('  note: BPE-16k perplexities are not comparable to published word-level '
                       'WikiText numbers  |  '
                       f'watching {_fmt(time.time()-t0)}  |  read-only, Ctrl+C safe', style='dim'))
    return Group(*blocks)


def _total_epochs(tag):
    """Read the real epoch count for the current run, robust to stale content in a reused log.
    Prefer the live tqdm tail (cannot be stale), then the newest header line, then the default."""
    p = os.path.join(LOGS, f'{tag}.log')
    total = None
    try:
        with open(p, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8000))
            tail = f.read().decode('utf-8', 'replace').replace('\r', '\n')
        for line in reversed(tail.split('\n')):
            m = re.search(r'epoch\s+\d+/(\d+)', line)
            if m:
                return int(m.group(1))
        with open(p, errors='replace') as f:
            for line in f:
                m = re.search(r'(\d+) epochs, batch', line)
                if m:
                    total = int(m.group(1))
    except OSError:
        pass
    return total or 30


def _run_meta(tag):
    """Parse which GPU a run is on, its parameter count, and its dataset from its log header.
    Last match wins, so a stale header in a reused log never beats the current run's."""
    p = os.path.join(LOGS, f'{tag}.log')
    gpu, params, dataset = None, None, None
    try:
        with open(p, errors='replace') as f:
            for line in f:
                m = re.search(r'device cuda:(\d+)', line)
                if m:
                    gpu = int(m.group(1))
                m = re.search(r'([\d,]+) parameters', line)
                if m:
                    params = int(m.group(1).replace(',', ''))
                m = re.search(r'dataset (\w+)\s+\|', line)
                if m:
                    dataset = m.group(1)
    except OSError:
        pass
    return gpu, params, dataset


def _fmt(s):
    s = int(s)
    if s < 60:
        return f'{s}s'
    if s < 3600:
        return f'{s//60}m'
    return f'{s//3600}h{(s%3600)//60:02d}m'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--interval', type=float, default=2.0)
    ap.add_argument('--live', action='store_true',
                    help='show only runs with a process still attached -- the usual case when '
                         'you are watching something rather than reviewing everything')
    ap.add_argument('--tag', default=None,
                    help='substring filter on the run name, e.g. --tag afriberta')
    ap.add_argument('--hours', type=float, default=18.0,
                    help='how far back to include finished runs (default 18)')
    a = ap.parse_args()

    global ONLY_LIVE, TAG_FILTER, RECENT_HOURS
    ONLY_LIVE, TAG_FILTER, RECENT_HOURS = a.live, a.tag, a.hours

    t0 = time.time()
    console = Console(legacy_windows=False)
    if a.once:
        console.print(render(t0))
        return
    # auto_refresh off, exactly one repaint per actual update -- the a1 flicker fix, kept.
    with Live(render(t0), console=console, auto_refresh=False, screen=False) as live:
        try:
            while True:
                time.sleep(a.interval)
                live.update(render(t0), refresh=True)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
