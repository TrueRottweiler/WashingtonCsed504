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
import hashlib
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
# A cell_tag-shaped name: corpus_<tokens>_<steps>[_preset]_s<seed>. These are the data-ladder
# grid cells; anything else is a differently-named experiment or a piece of scratch.
LADDER_TAG = re.compile(r'^[a-z]+_[\d.]+[kM]?_[\d.]+[kM]?(_[a-z]+)?_s\d+$')

# Throwaway runs, excluded from every comparison. An inclusion regex was the wrong shape: it
# silently dropped whole experiments -- the five-language comparison never appeared on the chart
# because `multi_eng` does not look like a grid cell.
SCRATCH = re.compile(r'^(teetest|resumetest|lrfix|warmupfix|tmp|smoke-)')


def experiment_of(tag: str) -> str:
    """Which experiment a run belongs to, from its name.

    Runs are compared against their own experiment, not against everything ever run. The data
    ladder and the five-language comparison answer different questions and putting them on one
    axis produces a chart that answers neither.
    """
    if LADDER_TAG.match(tag):
        return 'ladder'
    return tag.split('_')[0]

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


def live_from_files(window_s: float = 210.0) -> set:
    """Tags whose curve file was written to in the last few minutes.

    This is the reliable half of liveness detection and it exists because the other half keeps
    failing. `live_runs()` below finds runs by matching a process command line, which only works
    for runs launched as their own `mlm_run.py` process. Studies now call `mlm_api.pretrain()`
    IN-PROCESS, so a study spending fifty minutes pretraining shows up nowhere in the process
    scan -- the dashboard reported "0 running" while a card sat at 94% and 300 watts, and it was
    not wrong so much as looking for the wrong thing.

    A training run appends to runs/<tag>.jsonl every logging interval whatever launched it. File
    mtime is therefore the one signal that does not care how the run was started. The window is
    generous because the logging interval on a slow cell is a couple of minutes.
    """
    now = time.time()
    out = set()
    for p in glob.glob(os.path.join(RUNS, '*.jsonl')):
        try:
            if now - os.path.getmtime(p) < window_s:
                out.add(os.path.basename(p)[:-6])
        except OSError:
            pass
    return out


def running_studies() -> dict:
    """Study scripts with a live process, mapped to the GPU they were given.

    This is the answer to "what is training right now" for every study that fine-tunes, and there
    was never a good reason not to ask. A fine-tuning study writes nothing between cells, so run
    files cannot see it and the panel reported an idle machine — while the process was sitting in
    the process table the entire time with its name and its --gpu argument on the command line.

    Saying "I cannot account for this work" is the right answer only when it is true. It was not.
    """
    try:
        out = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
             r"Where-Object { $_.CommandLine -match 'study_\w+\.py' } | "
             "ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True, timeout=8).stdout
    except Exception:                                  # noqa: BLE001
        return {}
    found = {}
    for line in out.splitlines():
        m = re.search(r'(study_\w+\.py)', line)
        if not m:
            continue
        script = m.group(1)
        # The card comes from CUDA_VISIBLE_DEVICES when the launcher set it on the command line,
        # and from --gpu otherwise. Before #64 --gpu was ignored by every fine-tuning study, so a
        # launcher that wanted card 1 had to set the variable instead; the flag works now, and
        # --gpu is the form to prefer because it is visible here. When the variable is exported by
        # the shell rather than written inline it is invisible to a command-line scan, and the
        # honest answer is None rather than a confident wrong number.
        env = re.search(r'CUDA_VISIBLE_DEVICES=(\d+)', line)
        gpu = re.search(r'--gpu\s+(\d+)', line)
        card = int(env.group(1)) if env else (int(gpu.group(1)) if gpu else None)

        # Key on script AND shard. Two shards of one study are two processes with the same script
        # name, and keying on the name alone collapsed them into one -- so the panel showed a
        # single worker while both cards were at 70%, which is the failure this function exists
        # to prevent, one level up.
        shard = re.search(r'--shard\s+(\S+)', line)
        key = f'{script} {shard.group(1)}' if shard else script
        found[key] = {'gpu': card, 'script': script,
                      'shard': shard.group(1) if shard else None}
    return found


def cards_busy(threshold: int = 25) -> list:
    """Utilization per GPU, so the page can notice work it cannot otherwise account for."""
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=8).stdout
    except Exception:                                  # noqa: BLE001 - no GPU, or no driver
        return []
    busy = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) >= 2 and parts[1].isdigit() and int(parts[1]) >= threshold:
            busy.append({'index': int(parts[0]), 'util': int(parts[1]),
                         'mem_mb': int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0})
    return busy


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


# Corpus code -> what a person calls it. Anything unlisted falls back to the code itself, so a
# new language shows up readable-ish rather than breaking the description.
LANGUAGES = {
    'eng': 'English', 'eng_1b': 'English', 'fra': 'French', 'ind': 'Indonesian', 'cmn': 'Mandarin',
    'yor': 'Yoruba', 'ibo': 'Igbo',
    'swh': 'Swahili', 'hau': 'Hausa', 'amh': 'Amharic', 'afr': 'Afrikaans',
    'som': 'Somali', 'xho': 'Xhosa', 'kin': 'Kinyarwanda', 'sna': 'Shona',
    'lug': 'Luganda', 'wol': 'Wolof', 'nya': 'Chichewa',
    'wikitext2': 'WikiText-2', 'wikitext103': 'WikiText-103', 'shakespeare': 'Shakespeare',
}

# Preset -> the number people actually compare models by.
PRESET_SIZE = {'poc': '33.8M', 'afriberta': '86M'}


def _compact_n(n):
    """4,000,000 -> 4M. Local copy so this module does not depend on the trainer being importable."""
    if not n:
        return None
    for unit, size in (('M', 1_000_000), ('k', 1_000)):
        if n >= size:
            return f'{n / size:.1f}'.rstrip('0').rstrip('.') + unit
    return str(n)


def _tokens_per_step(meta, cli, result):
    """batch x seq_len, from whichever record happens to carry them."""
    for src in (meta, cli, result or {}):
        b, sl = src.get('batch'), src.get('seq_len')
        if b and sl:
            return b * sl
    return None


def _passes(rows, meta, cli, result):
    """How many times the model has seen the corpus, as of the last logged point.

    Read from the record where it is present and derived where it is not, because every run
    started before this field existed logs only steps -- and a stat that is blank for the run you
    are actually watching is worse than no stat at all.
    """
    if rows and rows[-1].get('passes') is not None:
        return rows[-1]['passes']
    tps = _tokens_per_step(meta, cli, result)
    n = meta.get('n_tokens') or (result or {}).get('n_tokens')
    if not (rows and tps and n):
        return None
    return rows[-1].get('step', 0) * tps / n


def _total_passes(rows, meta, cli, result):
    """How many passes the whole run will make."""
    if rows and rows[-1].get('total_passes') is not None:
        return rows[-1]['total_passes']
    tps = _tokens_per_step(meta, cli, result)
    n = meta.get('n_tokens') or (result or {}).get('n_tokens')
    steps = meta.get('steps') or cli.get('steps') or (result or {}).get('steps')
    if not (tps and n and steps):
        return None
    return steps * tps / n


def describe_run(tag, corpus, n_tokens, preset, steps, batch, seed, params=None):
    """One human sentence for a run: what language, how much text, what model, how much training.

    Tags are built for uniqueness and sorting, which makes them unreadable -- nobody can look at
    `yor_16M_11.7k_afriberta_s0` and say what it is testing. This says it.
    """
    lang = LANGUAGES.get(corpus, corpus or 'unknown corpus')
    size = PRESET_SIZE.get(preset or 'poc')
    if not size and params:
        size = _compact_n(params)

    bits = [lang]
    if n_tokens:
        bits.append(f'{_compact_n(n_tokens)} tokens of text')
    if size:
        bits.append(f'{size} model')
    if steps:
        bits.append(f'{_compact_n(steps)} steps')
    if batch and batch != 128:
        bits.append(f'batch {batch}')
    if seed:
        bits.append(f'seed {seed}')
    return ' \u00b7 '.join(bits)


_ENTROPY_CACHE = {}


def unigram_entropy(corpus: str) -> float | None:
    """Cross-entropy a model would score by predicting token frequencies and ignoring context.

    This is the anchor that makes two languages comparable. Raw validation loss is not: each
    corpus has its own 16k vocabulary with its own frequency distribution, so a loss of 2.5 on
    French and 4.5 on Mandarin are not measurements of the same thing. Most of that gap is the
    vocabulary, not the language. Subtracting this leaves what the model learned FROM CONTEXT,
    which is comparable.

    Cached to disk -- it reads twenty million tokens and never changes for a prepared corpus.
    """
    if corpus in _ENTROPY_CACHE:
        return _ENTROPY_CACHE[corpus]

    cache_file = os.path.join(HERE, 'data', corpus, 'unigram_entropy.json')
    try:
        with open(cache_file, encoding='utf-8') as f:
            _ENTROPY_CACHE[corpus] = json.load(f)['entropy']
            return _ENTROPY_CACHE[corpus]
    except (OSError, ValueError, KeyError):
        pass

    try:
        import numpy as np
        path = os.path.join(HERE, 'data', corpus, 'train_tokens.npy')
        arr = np.load(path, mmap_mode='r')[:20_000_000]
        counts = np.bincount(np.asarray(arr, dtype=np.int64)).astype(np.float64)
        p = counts / counts.sum()
        p = p[p > 0]
        h = float(-(p * np.log(p)).sum())
    except Exception:                                  # noqa: BLE001 -- no corpus, no anchor
        _ENTROPY_CACHE[corpus] = None
        return None

    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({'entropy': h, 'tokens_counted': 20_000_000}, f)
    except OSError:
        pass
    _ENTROPY_CACHE[corpus] = h
    return h


def measured_rates() -> dict:
    """Tokens of updates per second, per preset, from every run this machine has finished.

    Estimates come from this machine's own history rather than a constant in the source. A
    hard-coded rate is wrong on any other machine and goes quietly stale on this one the moment
    a driver or a batch size changes.
    """
    rates = {}
    for rp in glob.glob(os.path.join(RUNS, '*_result.json')):
        try:
            with open(rp, encoding='utf-8') as f:
                r = json.load(f)
        except (OSError, ValueError):
            continue
        steps, batch = r.get('steps'), r.get('batch')
        secs = r.get('elapsed_sec') or r.get('seconds') or r.get('wall_sec')
        if not (steps and batch and secs):
            continue
        preset = r.get('preset') or 'poc'
        got = rates.setdefault(preset, [0, 0.0])
        got[0] += steps * batch * (r.get('seq_len') or 128)
        got[1] += secs
    return {p: tok / sec for p, (tok, sec) in rates.items() if sec > 0}


def schedule_remaining(cells, n_gpu, rates, live):
    """Wall-clock left, by simulating the same longest-first fill the fleet actually uses.

    Dividing the remaining work by the number of cards is wrong whenever the cells differ in
    cost: at the end of a queue the cards drain unevenly, and the finish time is set by the
    slowest single remaining cell, not by the average.
    """
    default = rates.get('poc') or 400_000.0
    cards = [0.0] * max(1, n_gpu)

    # Cells already running occupy their card for whatever is left of them.
    for c in cells:
        if c['state'] != 'running':
            continue
        frac = live.get(c['tag']) or 0.0
        cards[cards.index(min(cards))] += (c.get('run_s') or 0) * max(0.0, 1 - frac)

    todo = sorted((c for c in cells if c['state'] == 'pending'),
                  key=lambda c: -c['update_tokens'] / (rates.get(c['preset']) or default))
    for c in todo:
        i = cards.index(min(cards))
        cards[i] += c.get('run_s') or 0
    return max(cards) if cards else 0.0


def _prefix_of(tag: str, corpus: str | None) -> str:
    """The tag's namespace, rendered for a human: "lr15 · ".

    Two configurations that differ only in learning rate or clipping produce the same
    description -- same corpus, same size, same steps -- and a queue listing six of them reads as
    one experiment repeated. The prefix is the only thing that distinguishes them, so it belongs
    in the label rather than only in the tag.
    """
    if not corpus or not tag or tag.startswith(corpus + '_'):
        return ''
    head = tag.split(corpus + '_')[0].rstrip('_')
    return f'{head} · ' if head else ''


# Cells announced before fleet_plan stamped attribution, and cells announced by a study that is
# ALREADY RUNNING with the older module loaded, carry no 'study'. Rather than wait for every
# long-running script to be restarted, work it out from the tag. This is a fallback and it is
# meant to age out: anything it cannot place says so rather than guessing.
_STUDY_BY_PREFIX = (
    ('ft_masakhaner_yor_yor-random-init',       'MasakhaNER: sweeping the untrained floor',    'Patrick'),
    ('ft_masakhaner_yor_xlm-roberta-base-random-init',
                                                'MasakhaNER: sweeping the untrained floor',    'Patrick'),
    ('ft_sib200_yor_Latn_swap',                 'tokenizer swap: both arms, rate on dev',      'Patrick'),
    ('ft_masakhaner_yor_swap',                  'tokenizer swap: both arms, rate on dev',      'Patrick'),
    ('swap_yor_xlmr',                           'tokenizer penalty: three more seeds per arm', 'Patrick'),
    ('swap62k',                                 'tokenizer penalty: three more seeds per arm', 'Patrick'),
    ('clipprev',                                'clipping: does it prevent failure',           'Jeffrey'),
    ('lrx',                                     'learning-rate transfer across languages',     'Jeffrey'),
)


def attribute(cell) -> tuple[str, str | None]:
    """Which study a cell belongs to, and whose it is."""
    if cell.get('study'):
        return cell['study'], cell.get('owner')
    tag = cell.get('tag', '')
    for prefix, study, owner in _STUDY_BY_PREFIX:
        if tag.startswith(prefix):
            return study, owner
    # Everything else is the downstream-correlation batch and other cells that predate
    # fleet_plan, declared by hand in declare_studies.py. Say that rather than "unattributed",
    # which reads like the dashboard is broken when it is just describing older work.
    return 'earlier work, declared before studies were labelled', None


def fleet_plan(runs, hours: float) -> dict | None:
    """The queued study, with each cell's status derived rather than recorded.

    mlm_fleet publishes only the plan. Status is worked out here from what is on disk and what
    is running, so the file cannot go stale between a cell finishing and something rewriting it
    -- and a fleet killed halfway leaves a plan that still reads correctly.
    """
    path = os.path.join(RUNS, '_fleet_plan.json')
    try:
        with open(path, encoding='utf-8') as f:
            plan = json.load(f)
    except (OSError, ValueError):
        return None

    live = {r['tag']: (r.get('frac') or 0.0) for r in runs if r.get('live')}
    rates = measured_rates()
    default = rates.get('poc') or 400_000.0

    cells, done, pending, running = [], 0, 0, 0
    for c in plan.get('cells', []):
        tag = c['tag']
        # A night is not only pretraining fleets. Fine-tuning work writes <tag>_ft.json, takes
        # minutes rather than hours, and was invisible here -- so a queue panel showing six cells
        # was hiding four more that a person waiting on the machine would want to see.
        kind = c.get('kind', 'pretrain')
        record = f'{tag}_ft.json' if kind == 'finetune' else f'{tag}_result.json'
        # A fine-tune in flight writes nothing until it finishes, so without this marker such a
        # cell goes QUEUED -> done with no visible middle and the panel reports zero running
        # through an entire sweep. ft_api.evaluate() writes the marker and removes it in a
        # finally block; anything older than ten minutes is treated as a crashed run's litter
        # rather than as work, so a killed study cannot leave a cell looking busy forever.
        marker = os.path.join(RUNS, f'{tag}_ft.running')
        in_flight = False
        try:
            in_flight = time.time() - os.path.getmtime(marker) < 600
        except OSError:
            pass

        if tag in live or in_flight:
            state, running = 'running', running + 1
        elif os.path.exists(os.path.join(RUNS, record)):
            state, done = 'done', done + 1
        else:
            state, pending = 'pending', pending + 1

        if kind == 'finetune':
            # Measured from the fine-tunes already on disk rather than guessed; they are all a
            # few minutes and the panel only needs the order of magnitude right.
            full = c.get('eta_s', 300)
        else:
            rate = rates.get(c['preset']) or default
            full = c['update_tokens'] / rate
        cells.append({**c, 'state': state,
                      'eta_s': None if state == 'done' else
                               full * (1 - live.get(tag, 0.0)) if state == 'running' else full,
                      'run_s': full,
                      'description': c.get('label') or (
                          _prefix_of(tag, c.get('corpus') or plan.get('corpus'))
                          + describe_run(tag, c.get('corpus') or plan.get('corpus'),
                                         c.get('tokens'), c.get('preset'), c.get('steps'),
                                         plan.get('batch'), c.get('seed')))})

    n_gpu = plan.get('n_gpu') or 1
    left = schedule_remaining(cells, n_gpu, rates, live)

    # One group per study, so the panel answers "whose work is the machine doing" rather than
    # only "is the machine busy". Groups with something running sort first, then groups with
    # work left, then the finished ones -- which is the order you care about when you look up
    # from something else and want to know whether to keep waiting.
    live_scripts = running_studies()
    groups = {}
    for c in cells:
        study, owner = attribute(c)
        c['study'], c['owner'] = study, owner
        g = groups.setdefault(study, {'study': study, 'owner': owner, 'cells': [],
                                      'done': 0, 'running': 0, 'pending': 0,
                                      'script': c.get('script'), 'active': False, 'gpu': None})
        g['cells'].append(c)
        g[c['state']] += 1
        if c.get('script') and not g.get('script'):
            g['script'] = c['script']

    # A study whose process is alive is working, whatever its individual cells look like. Mark
    # the group active and, when nothing else has claimed a cell, attribute the work to its next
    # pending one -- that is what a sequential study is doing by construction. It is flagged as
    # inferred rather than observed, because the honest version of "what is running" here is
    # "this study, somewhere in this cell", not a measurement of that cell.
    # live_scripts is keyed by script AND shard, so two shards of one study stay distinct there.
    # A plan group names only the script, so match on the script field inside each entry rather
    # than on the key -- and keep every worker that matches, because "how many cards is this
    # study using" is exactly what the panel should be able to say.
    matched = {}
    for key, info in live_scripts.items():
        for g in groups.values():
            if g.get('script') and g['script'] == info['script']:
                matched.setdefault(key, g)
                break

    # Plans written before announce() recorded its script have no name to match on, and a study
    # already running when this code landed will never write one. Rather than be blind until
    # every study restarts: if some study process is alive, is unmatched, and exactly one group
    # still has work outstanding, that is where the work is. One unexplained worker and one
    # unfinished queue is not a guess worth hedging over.
    if len(live_scripts) > len(matched):
        # Deliberately NOT requiring pending cells. A study whose declared cells are all done but
        # whose process is still alive is doing work it never declared -- which has happened, and
        # is exactly the case where saying so matters most. Requiring pending here made the panel
        # go quiet in the one situation it was rebuilt to catch.
        spare = [s for s in live_scripts if s not in matched]
        candidates = [g for g in groups.values() if not g.get('script')]
        # With every declared cell finished, "the group with work outstanding" no longer picks
        # one out, so fall back to the plan's own note of who announced last -- the running study
        # is overwhelmingly likely to be it. Only used when there is exactly one live script to
        # place; two unexplained processes and a guess is worse than saying nothing.
        if len(spare) == 1 and len(candidates) > 1:
            candidates = [g for g in candidates if g['study'] == plan.get('queue')]
        if len(candidates) == 1 and len(spare) == 1:
            # spare[0] is a "script shard" key; store the script name, which is what a plan
            # group carries and what the next poll will match on.
            candidates[0]['script'] = live_scripts[spare[0]]['script']
            candidates[0]['script_inferred'] = True
            matched[spare[0]] = candidates[0]

    for key, g in matched.items():
        g['active'] = True
        g.setdefault('workers', []).append(
            {'gpu': live_scripts[key].get('gpu'), 'shard': live_scripts[key].get('shard')})
        # One card in `gpu` for the header's simple case; `workers` carries the full picture when
        # a study is sharded across both.
        if g.get('gpu') is None:
            g['gpu'] = live_scripts[key].get('gpu')
        if not g['running'] and g['pending']:
            nxt = next(c for c in g['cells'] if c['state'] == 'pending')
            nxt['state'], nxt['inferred'] = 'running', True
            g['running'] += 1
            g['pending'] -= 1
            running += 1
            pending -= 1
    for g in groups.values():
        g['remaining_s'] = schedule_remaining(g['cells'], n_gpu, rates, live)
        g['finish_at'] = time.time() + g['remaining_s']
    ordered = sorted(groups.values(),
                     key=lambda g: (0 if g['running'] else 1 if g['pending'] else 2,
                                    -g['running'], -g['pending']))

    return {'corpus': plan.get('corpus'), 'queue': plan.get('queue'),
            'started': plan.get('started'), 'n_gpu': n_gpu, 'cells': cells,
            'groups': ordered,
            'done': done, 'running': running, 'pending': pending,
            'remaining_s': left, 'finish_at': time.time() + left,
            'rates': rates, 'measured_from': sum(1 for _ in glob.glob(
                os.path.join(RUNS, '*_result.json')))}


def snapshot(hours: float) -> dict:
    """Everything the page draws, in one JSON payload."""
    alive = live_runs()
    # A run that is writing its curve right now is running, whatever the process scan thinks.
    # The two sources disagree exactly when a study pretrains in-process, which is most of them
    # now, so the union is what the page should believe.
    for tag in live_from_files():
        alive.setdefault(tag, {})
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
            'unigram_h': unigram_entropy(corpus),
            'description': describe_run(
                tag, corpus, meta.get('n_tokens') or (result or {}).get('n_tokens'),
                meta.get('preset') or cli.get('preset') or (result or {}).get('preset'),
                step_total, meta.get('batch') or cli.get('batch') or (result or {}).get('batch'),
                meta.get('seed'), meta.get('params')),
            'step': last.get('step'), 'steps': step_total,
            'train_loss': last['train']['loss'], 'val_loss': last['val']['loss'],
            'random_loss': random_loss, 'lr': last.get('lr'),
            'elapsed': elapsed, 'since_point': since, 'log_every': meta.get('log_every'),
            'tok_s': med_tps, 'eta_s': eta,
            'corpus': corpus,
            'steps_total': step_total,
            'study': not bool(SCRATCH.match(tag)),
            'experiment': experiment_of(tag),
            'n_tokens': meta.get('n_tokens') or (result or {}).get('n_tokens'),
            'seed': meta.get('seed'),
            'accum': meta.get('accum'),
            'preset': (meta.get('preset') or cli.get('preset')
                       or (result or {}).get('preset')),
            'batch': (meta.get('batch') or cli.get('batch')
                      or (result or {}).get('batch')),
            'stalled': (result or {}).get('stalled'),
            'passes': _passes(rows, meta, cli, result),
            'total_passes': _total_passes(rows, meta, cli, result),
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
    fleet = fleet_plan(runs, hours)

    # Reconciliation. Everything above this line trusts what studies declared about themselves;
    # this compares that against the one thing that cannot be misdeclared, which is whether the
    # cards are drawing power. Four separate times this panel has said "finished" while the
    # machine was busy, and each time the fix was to declare better -- which works until the next
    # study declares late, or declares a cell under a name the panel does not recognise.
    #
    # So: when the GPUs are working and nothing in the plan accounts for it, say exactly that.
    # A dashboard that admits it cannot explain the machine is far more useful than one that
    # quietly reports zero. This turns the failure that keeps recurring from invisible into loud,
    # which is the only property that has ever actually held.
    busy = cards_busy()
    studies = running_studies()
    # Only genuinely unexplained if no run is writing a curve AND no study process is alive. The
    # first version fired on the second condition alone and told the reader nothing was known,
    # while the answer was one process scan away.
    unexplained = bool(busy) and not any(r['live'] for r in runs) and not studies
    return {'now': time.time(), 'server_started': SERVER_STARTED,
            'fleet': fleet,
            'page_version': page_version(),
            'busy_cards': busy, 'unexplained_work': unexplained,
            'running_studies': [{'script': k, **v} for k, v in sorted(studies.items())],
            'gpus': gpus(), 'runs': runs}


def page_version() -> str:
    """A short hash of the page we would serve right now.

    The dashboard's JavaScript is embedded in the HTML, so editing this file changes nothing in
    a browser tab that is already open -- it keeps polling the API and rendering with its old
    code. That has been mistaken for a broken feature three times. The page compares this stamp
    against the one it was built with and tells the reader to reload when they differ.
    """
    return hashlib.sha256(PAGE.encode('utf-8')).hexdigest()[:8]


class Handler(BaseHTTPRequestHandler):
    hours = 18.0

    def log_message(self, *a):
        pass                      # a dashboard should not spam the console it runs in

    def do_GET(self):
        if self.path.startswith('/api/'):
            body = json.dumps(snapshot(self.hours)).encode()
            ctype = 'application/json'
        else:
            body = PAGE.replace('__PAGE_VERSION__', page_version()).encode()
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
#queue{margin:0 0 1rem}
.qgroup{border-top:1px solid var(--line)}
.qgroup:first-child{border-top:0}
.qghead{display:grid;grid-template-columns:5.5rem 1fr auto;gap:.7rem;align-items:baseline;
  padding:.5rem 1.25rem .42rem;font-size:.82rem}
.qgroup.live .qghead{background:color-mix(in oklab,var(--blue) 9%,transparent)}
/* Whose work it is, as its own column rather than buried in the study name -- the panel is read
   from across the room and the owner is the thing being looked for. */
.qgowner{font-size:.7rem;text-transform:uppercase;letter-spacing:.05em;color:var(--dim)}
.qgroup.live .qgowner{color:var(--blue)}
.qgname{font-weight:600}
.qgcount{color:var(--dim);font-variant-numeric:tabular-nums;font-size:.76rem}
.qrow{display:grid;grid-template-columns:1.1rem 1fr auto auto auto;gap:.7rem;align-items:center;
  padding:.32rem 1.25rem;font-size:.82rem;border-top:1px solid var(--line)}
.qrow:first-child{border-top:0}
.qrow.pending{opacity:.6}
.qdot{width:.6rem;height:.6rem;border-radius:50%;background:var(--dim)}
.qrow.running .qdot{background:var(--blue);
  box-shadow:0 0 0 3px color-mix(in oklab,var(--blue) 25%,transparent)}
.qrow.done .qdot{background:var(--aqua)}
.qnum{color:var(--dim);font-variant-numeric:tabular-nums;font-size:.76rem;min-width:5.6rem;
  text-align:right}
.qstate{font-size:.7rem;text-transform:uppercase;letter-spacing:.03em;color:var(--dim);
  min-width:4.4rem;text-align:right}
.qfoot{padding:.5rem 1.25rem .7rem;font-size:.76rem;border-top:1px solid var(--line)}
.stale{padding:.6rem 1.25rem;background:color-mix(in oklab,var(--amber) 18%,var(--bg));
  border-bottom:1px solid color-mix(in oklab,var(--amber) 45%,var(--line));font-size:.82rem}
.stale b{color:var(--ink)}
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
.takeaway{margin:.1rem 1.25rem .5rem;font-size:.84rem;color:var(--ink);line-height:1.45}
.takeaway em{font-style:normal;font-weight:650}
.sel{font-size:.76rem;color:var(--dim);display:inline-flex;gap:.35rem;align-items:center}
.sel select{font:inherit;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:5px;padding:.15rem .3rem}
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
table.past td:first-child{word-break:normal;min-width:190px}
.desc{font-size:.76rem;color:var(--dim);font-weight:400;margin-top:.1rem}
.tagsm{font-size:.66rem;color:var(--dim);opacity:.65;margin-top:.1rem;word-break:break-all}
/* Half the live card's chart, so a table of them stays a table. */
td.mini-cell{width:270px;padding:.15rem .5rem}
svg.mini{width:260px;height:66px;display:block}
@media (max-width:800px){td.mini-cell{display:none}}
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
<section id="queue" class="card" hidden>
  <div class="chd"><h2 id="qtitle">Queued study</h2>
    <span class="muted" id="qnote"></span></div>
  <div id="qbody"></div>
  <div class="qfoot muted" id="qfoot"></div>
</section>
<div id="stale" class="stale" hidden>
  This page is running older code than the server. <b>Reload</b> to pick up the latest.
</div>
<div id="unexplained" class="stale" hidden></div>
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
    <div class="chd"><h2 id="cmptitle">All runs together</h2>
      <label class="sel">experiment
        <select id="experiment"></select></label>
      <label class="sel">compare by
        <select id="axis">
          <option value="auto">what varies</option>
          <option value="corpus">language</option>
          <option value="n">data size</option>
          <option value="preset">model size</option>
          <option value="steps">compute</option>
        </select></label>
      <label class="sel">measure
        <select id="measure">
          <option value="loss">validation loss</option>
          <option value="gain">context gained (comparable across languages)</option>
        </select></label>
      <span class="muted" id="cmpnote"></span></div>
    <p class="takeaway" id="takeaway"></p>
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
  const W=520,H=146,L=42,R=10,T=10,B=36;
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
  // Five evenly spaced ticks across the full budget, with a mark on the axis rather than a
  // bare number. Previously the only labels were the first point, the last point and the total
  // -- and on a run that is nearly finished the last two land on top of each other, which is
  // what produced the "12k steps12k" overlap.
  let xticks = '';
  const NT = 5;
  for(let i=0;i<=NT;i++){
    const v = x0 + (x1-x0)*i/NT;
    const px = X(v);
    const anchor = i===0 ? 'start' : (i===NT ? 'end' : 'middle');
    xticks += `<line x1="${px}" y1="${H-B}" x2="${px}" y2="${H-B+3}" class="axis"/>`
            + `<text class="tk" x="${px}" y="${H-B+16}" text-anchor="${anchor}">${fmtN(v)}</text>`;
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
    ${xticks}
    <text class="tk" x="${(L+W-R)/2}" y="${H-B+31}" text-anchor="middle"
      >optimizer step</text>
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

// A half-size curve for a finished run. Same encoding as the live card -- blue val, orange
// train, dashed "learned nothing" -- so a glance at the table reads the same way as a glance at
// a card. It answers the question a final number cannot: did this converge, or was it still
// falling when the budget ran out?
function miniChart(curve, randomLoss){
  if(!curve || curve.length < 2) return '';
  const W=260, H=66, L=4, R=4, T=6, B=6;
  const xs = curve.map(p=>p.x), ys = curve.flatMap(p=>[p.train, p.val]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys.concat(randomLoss ? [randomLoss] : []));
  const pad = (y1-y0)*0.08 || 0.1; y0 -= pad; y1 += pad;
  const X = v => L + (v-x0)/((x1-x0)||1) * (W-L-R);
  const Y = v => T + (y1-v)/((y1-y0)||1) * (H-T-B);
  const path = k => curve.map((p,i)=>(i?'L':'M')+X(p.x).toFixed(1)+' '+Y(p[k]).toFixed(1)).join(' ');
  const ref = (randomLoss && randomLoss<=y1 && randomLoss>=y0)
    ? `<line x1="${L}" y1="${Y(randomLoss)}" x2="${W-R}" y2="${Y(randomLoss)}"
         stroke="var(--dim)" stroke-width="1" stroke-dasharray="3 3" opacity=".6"/>` : '';
  const last = curve[curve.length-1];
  return `<svg class="mini" viewBox="0 0 ${W} ${H}" role="img"
      aria-label="convergence curve">${ref}
    <path d="${path('train')}" fill="none" stroke="var(--orange)" stroke-width="1.3"
      stroke-linejoin="round" opacity=".75"/>
    <path d="${path('val')}" fill="none" stroke="var(--blue)" stroke-width="1.7"
      stroke-linejoin="round"/>
    <circle cx="${X(last.x)}" cy="${Y(last.val)}" r="2.4" fill="var(--blue)"/></svg>`;
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
    <div class="rhead"><div><span class="tag">${r.tag}</span><div class="desc">${r.description||''}</div></div><span>${stall}${state}</span></div>
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
        <div class="v">${fmtN(r.step)}${r.steps?'<span class="sub"> / '+fmtN(r.steps)+'</span>':''}</div>
        <div class="sub">optimizer updates</div></div>
      <div><div class="k">epochs</div>
        <div class="v">${fmtPasses(r.passes)}${r.total_passes==null?'':'<span class="sub"> / '+fmtPasses(r.total_passes)+'</span>'}</div>
        <div class="sub">passes over the data</div></div>
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
// two model sizes read as a pair rather than as neighbors, and every line on the right is named
// in a legend. Ad-hoc runs are excluded -- a probe sitting between two grid cells is what made
// the first attempt unreadable.
const HUES = ["#3987e5","#199e70","#9085e9","#d9a441","#d95926","#e66767"];
const SIZES = {poc: "33.8M", afriberta: "86M"};
const isBig = r => (r.preset || 'poc') !== 'poc';
const fmtPasses = p => p == null ? "\u2013"
  : p >= 100 ? Math.round(p).toString()
  : p >= 10 ? p.toFixed(1) : p.toFixed(2);
const fmtTok = n => n >= 1e6 ? (n/1e6).toFixed(0)+'M' : fmtN(n);

// One row per (data rung x model size), seeds averaged, so three seeds of one cell are one bar
// rather than three bars pretending to be different configurations.
function cells(runs, axis){
  // Group by whatever dimension the comparison is about. The panel used to key on data size
  // alone, which is right for a data ladder and useless for a set of runs that share their data
  // size and differ by language -- they all collapsed into one indistinguishable group.
  const grid = {};
  for(const r of runs){
    if(!r.study || r.val_loss == null) continue;
    // A cell is a run name with the seed stripped. Keying on (corpus, tokens, preset, steps)
    // instead looked principled and merged runs that share those but differ in ways the API does
    // not model -- the causal baselines carry none of those four, so GPT and LSTM at wikitext103
    // collapsed into a single averaged bar. The name is the one thing that always distinguishes
    // a configuration, and stripping the trailing _s<n> is exactly what makes seeds of one cell
    // group together and nothing else.
    const key = r.tag.replace(/_s\d+$/, '');
    (grid[key] = grid[key] || {base: key, corpus: r.corpus, n: r.n_tokens,
                               preset: r.preset || 'poc', steps: r.steps_total || 0,
                               runs: []}).runs.push(r);
  }
  return Object.values(grid).map(c => {
    const v = c.runs.map(r => r.val_loss);
    c.val = v.reduce((a,b)=>a+b,0)/v.length;
    c.seeds = v.length;
    c.live = c.runs.some(r => r.live);
    c.curve = c.runs.slice().sort((a,b)=>b.curve.length-a.curve.length)[0].curve;
    c.random = c.runs.map(r=>r.random_loss).find(Boolean);
    c.unigram = c.runs.map(r=>r.unigram_h).find(Boolean);
    // How much better than frequency-guessing. The only measure that survives a change of
    // language, because it cancels the vocabulary's own entropy.
    c.gain = c.unigram ? c.unigram - c.val : null;
    c.group = axis === 'corpus' ? c.corpus
            : axis === 'preset' ? c.preset
            : axis === 'steps'  ? c.steps
            : c.n;
    return c;
  }).sort((a,b) => (a.group > b.group ? 1 : a.group < b.group ? -1 : 0) || a.val - b.val);
}

// Which dimension actually varies? Pick the one with the most distinct values, so a set of runs
// that differ only by language groups by language without being told.
function autoAxis(runs){
  const st = runs.filter(r=>r.study);
  const counts = {corpus:new Set(), n:new Set(), preset:new Set(), steps:new Set()};
  st.forEach(r=>{
    counts.corpus.add(r.corpus); counts.n.add(r.n_tokens);
    counts.preset.add(r.preset || 'poc'); counts.steps.add(r.steps_total);
  });
  let best = 'n', bestN = 0;
  for(const k of ['corpus','n','preset','steps']){
    if(counts[k].size > bestN){ bestN = counts[k].size; best = k; }
  }
  return best;
}

function rungColour(rungs){
  const m = {}; rungs.forEach((n,i)=> m[n] = HUES[i % HUES.length]); return m;
}

// LEFT: where each one lands. Grouped bars, x ordered by how much unique text the run saw.
function barsChart(cs, col, rungs, axis, usingGain){
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
    const inRung = cs.filter(c=>c.group===n);
    const bw = Math.min(30, (slot-16)/Math.max(inRung.length,1));
    inRung.forEach((c,bi)=>{
      const x = L + gi*slot + (slot - bw*inRung.length)/2 + bi*bw;
      const y = Y(c.plot);
      bars += `<rect x="${x}" y="${y}" width="${bw-3}" height="${Math.max(1,(H-B)-y)}" rx="2"
          fill="${col[c.key]}" opacity="${isBig(c)?0.45:1}"/>
        <text class="tk" x="${x+(bw-3)/2}" y="${y-4}" text-anchor="middle"
          style="font-weight:600">${c.plot.toFixed(2)}</text>`;
    });
    bars += `<text class="tk" x="${L+gi*slot+slot/2}" y="${H-B+15}" text-anchor="middle"
        style="font-weight:600">${axis==='corpus'?(LANG_NAMES[n]||n):(axis==='n'?fmtTok(n):n)}</text>`;
  });
  const ref = rl ? `<line x1="${L}" y1="${Y(rl)}" x2="${W-R}" y2="${Y(rl)}" stroke="var(--dim)"
      stroke-dasharray="4 4" stroke-width="1"/>
      <text class="tk" x="${W-R}" y="${Y(rl)-4}" text-anchor="end">learned nothing (${rl.toFixed(1)})</text>` : '';
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="final loss by data rung">
    ${g}${ref}${bars}<line class="axis" x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}"/>
    <text class="tk" x="4" y="${T+4}">${usingGain?'context gained':'val loss'}</text></svg>`;
}

// RIGHT: how each one gets there. Same colors, dash = larger model, dot = stopped improving.
function curvesChart(cs, col, usingGain){
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
    lines += `<path d="${d}" fill="none" stroke="${col[c.key]}" stroke-width="${c.live?2.8:2}"
        stroke-linejoin="round" stroke-dasharray="${isBig(c)?'5 4':'0'}"/>`;
    const f=flatIdx(pts);
    lines += `<circle cx="${X(pts[f].x)}" cy="${Y(pts[f].val)}" r="3.5" fill="${col[c.key]}"/>`;
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

// How many distinct values each dimension takes in this experiment. An axis with one value
// groups every run together: identical colors, identical legend entries, and a takeaway that
// reads "50M tokens ahead of 50M tokens". The dropdown should not offer it in the first place.
function axisSpread(runs){
  const st = runs.filter(r=>r.study && r.val_loss!=null);
  const v = {corpus:new Set(), n:new Set(), preset:new Set(), steps:new Set()};
  st.forEach(r=>{
    v.corpus.add(r.corpus); v.n.add(r.n_tokens);
    v.preset.add(r.preset || 'poc'); v.steps.add(r.steps_total);
  });
  return {corpus:v.corpus.size, n:v.n.size, preset:v.preset.size, steps:v.steps.size};
}

// Rebuild the axis dropdown for the experiment on screen, disabling the dimensions it holds
// constant. Returns the axis to actually use, falling back to auto when the current choice has
// become degenerate -- which happens simply by switching experiments.
function syncAxisOptions(runs){
  const spread = axisSpread(runs);
  const sel = document.getElementById('axis');
  const opts = [['auto','what varies']].concat(
    [['corpus','language'],['n','data size'],['preset','model size'],['steps','compute']]
      .map(([k,label]) => [k, spread[k] > 1 ? label : label + ' \u2014 same for every run']));
  const sig = opts.map(([k]) => k + (spread[k] > 1 ? '1' : '0')).join(',');
  if(sel.dataset.sig !== sig){
    const keep = sel.value;
    sel.innerHTML = opts.map(([k,label]) =>
      `<option value="${k}"${k!=='auto' && spread[k]<2 ? ' disabled' : ''}>${label}</option>`
    ).join('');
    sel.value = (keep && (keep === 'auto' || spread[keep] > 1)) ? keep : 'auto';
    sel.dataset.sig = sig;
  }
  if(sel.value !== 'auto' && spread[sel.value] < 2) sel.value = 'auto';
  return sel.value === 'auto' ? autoAxis(runs) : sel.value;
}

// The queue, including what has not started. Watching two live runs tells you nothing about
// whether they are the whole study or the first tenth of it.
function renderQueue(f){
  const el = document.getElementById('queue');
  el.hidden = !f || !f.cells || !f.cells.length;
  if(el.hidden) return;

  const groups = f.groups && f.groups.length ? f.groups
                 : [{study: f.queue, owner: null, cells: f.cells, done: f.done,
                     running: f.running, pending: f.pending, remaining_s: f.remaining_s}];

  document.getElementById('qtitle').textContent =
    groups.length > 1 ? `${groups.length} studies queued`
                      : (EXP_NAMES[f.queue] || f.queue || 'Queued study');
  document.getElementById('qnote').textContent =
    `${f.done} of ${f.cells.length} done \u00b7 ${f.running} running \u00b7 `
    + `${f.pending} not started`
    + (f.remaining_s > 0 ? ` \u00b7 about ${fmtT(f.remaining_s)} left` : '');

  // A finished group collapses to its header. Two hundred done rows push the running work off
  // the screen, and the running work is the reason anyone opened this.
  document.getElementById('qbody').innerHTML = groups.map(g => {
    const spent = g.running ? 'running now'
                : g.pending ? `${g.pending} queued`
                : 'finished';
    const rows = (g.running || g.pending)
      ? g.cells.filter(c => c.state !== 'done').map(c => `
        <div class="qrow ${c.state}">
          <span class="qdot"></span>
          <span>${c.description || c.tag}</span>
          <span class="qnum">${fmtTok(c.update_tokens)} updates</span>
          <span class="qnum">${c.state === 'running' ? fmtT(c.eta_s) + ' left'
                                                     : fmtT(c.run_s)}</span>
          <span class="qstate">${c.state === 'pending' ? 'queued' : c.state}</span>
        </div>`).join('')
      : '';
    return `
      <div class="qgroup ${g.running ? 'live' : ''}">
        <div class="qghead">
          <span class="qgowner">${g.owner ? g.owner : '\u2014'}</span>
          <span class="qgname">${EXP_NAMES[g.study] || g.study}</span>
          <span class="qgcount">${g.done}/${g.cells.length} done \u00b7 ${spent}${
            g.remaining_s > 0 ? ' \u00b7 ~' + fmtT(g.remaining_s) + ' left' : ''}</span>
        </div>
        ${rows}
      </div>`;
  }).join('');

  // Say where the numbers come from. An estimate whose basis is invisible gets trusted when it
  // should not be, and ignored when it should not be.
  const rate = Object.entries(f.rates || {})
    .map(([p,v]) => `${SIZES[p]||p} ${Math.round(v/1000)}k tok/s`).join(' \u00b7 ');
  document.getElementById('qfoot').innerHTML =
    (f.remaining_s > 0
      ? `Expected to finish about <b>${new Date(f.finish_at*1000)
          .toLocaleTimeString([], {hour:'numeric', minute:'2-digit'})}</b>. ` : 'Finished. ')
    + `Estimated from this machine's own finished runs${rate ? ' \u2014 ' + rate : ''}, `
    + `scheduled across ${f.n_gpu} card${f.n_gpu===1?'':'s'}.`;
}

function renderCompare(allRuns){
  const el = document.getElementById('compare');

  // Which experiments exist, and which one are we looking at? Keep the user's choice across
  // refreshes -- the panel repaints every three seconds and resetting the selector each time
  // would make it unusable.
  const names = [...new Set(allRuns.filter(r=>r.study && r.val_loss!=null)
                                   .map(r=>r.experiment))].sort();
  const sel = document.getElementById('experiment');
  const want = sel.value && names.includes(sel.value) ? sel.value
             : (names.includes('multi') ? 'multi' : names[0]);
  if(sel.options.length !== names.length || sel.value !== want){
    sel.innerHTML = names.map(n=>`<option value="${n}">${EXP_NAMES[n]||n}</option>`).join('');
    sel.value = want;
  }
  const runs = allRuns.filter(r => r.experiment === want);
  const spread = axisSpread(runs);
  const axis = syncAxisOptions(runs);
  const measure = document.getElementById('measure').value;

  const cs = cells(runs, axis);
  el.hidden = cs.length < 2;
  if(el.hidden) return;

  const usingGain = measure === 'gain' && cs.every(c => c.gain != null);
  cs.forEach(c => { c.plot = usingGain ? c.gain : c.val; });

  const groups = [...new Set(cs.map(c=>c.group))]
      .sort((a,b) => (typeof a === 'number' ? a-b : String(a).localeCompare(String(b))));
  // Cells within a group still need to be told apart, so the hue is per cell and the group only
  // decides the ordering.
  const col = {};
  cs.forEach((c,i)=>{ c.key = c.base; col[c.key] = HUES[i % HUES.length]; });

  document.getElementById('bars').innerHTML   = barsChart(cs, col, groups, axis, usingGain);
  document.getElementById('curves').innerHTML = curvesChart(cs, col, usingGain);

  const AXIS_NAME = {corpus:'language', n:'unique tokens', preset:'model size', steps:'steps'};
  document.getElementById('lt').textContent = usingGain
    ? 'How much context each one learned  (higher is better)'
    : 'Where each one lands  (lower is better)';
  document.getElementById('rt').textContent =
    'How each one gets there  (dot = stopped improving)';
  const expSel = document.getElementById('experiment');
  document.getElementById('cmptitle').textContent =
    EXP_NAMES[expSel.value] || 'All runs together';
  document.getElementById('takeaway').innerHTML =
    (EXP_QUESTIONS[expSel.value] ? '<span class="muted">'
       + EXP_QUESTIONS[expSel.value] + '</span> ' : '')
    + takeaway(cs, spread, usingGain);

  document.getElementById('cmpnote').textContent =
    `${cs.length} configurations \u00b7 grouped by ${AXIS_NAME[axis]}`
    + (usingGain ? ' \u00b7 vocabulary entropy removed' : '');

  document.getElementById('cmpleg').innerHTML = cs.map(c =>
    '<span><i class="sw" style="background:' + col[c.key]
    + (isBig(c) ? ';opacity:.45' : '') + '"></i>' + cellLabel(c, spread)
    + (c.seeds > 1 ? ' (' + c.seeds + ' seeds)' : '') + '</span>').join('')
    + (usingGain ? '' : '<span style="opacity:.7">solid 33.8M &middot; dashed 86M</span>');
}

const EXP_QUESTIONS = {
  ladder: 'Does more Yoruba text help, if the compute is held fixed?',
  multi:  'At the same data and the same compute, how much does the language matter?',
  eng:    'With compute held fixed, does more text keep helping — and does the bigger '
          + 'model ever overtake the smaller one?',
  wikitext103: 'How do the older architectures do on the same English text?',
};

// One sentence a person can act on. The dashboard is for watching runs, not for analysis, so
// this says which configuration won and by how much -- and nothing else. The reasoning belongs
// in the reports.
function takeaway(cs, spread, usingGain){
  if(cs.length < 2) return '';
  const better = usingGain ? (a,b) => b.plot - a.plot : (a,b) => a.plot - b.plot;
  const rank = cs.slice().sort(better);
  const top = rank[0], bot = rank[rank.length-1];
  const gap = Math.abs(top.plot - bot.plot);
  const unit = usingGain ? 'nats of context' : 'nats of loss';
  if(gap < 0.10)
    return `<em>${cellLabel(top, spread)}</em> and <em>${cellLabel(bot, spread)}</em> land within `
         + `${gap.toFixed(2)} of each other \u2014 too close to call apart from seed noise.`;
  return `Best so far: <em>${cellLabel(top, spread)}</em> at ${top.plot.toFixed(2)}, `
       + `${gap.toFixed(2)} ${unit} ahead of <em>${cellLabel(bot, spread)}</em>.`;
}

// A cell's name is the set of dimensions the experiment varies. Labeling by the comparison
// axis alone produced five legend entries all reading "50M tokens" on an experiment where the
// data size is the one thing every run shares -- the label has to say what is DIFFERENT, which
// is not necessarily what is being grouped by.
function cellLabel(c, spread){
  const bits = [];
  if(spread.corpus > 1) bits.push(LANG_NAMES[c.corpus] || c.corpus);
  if(spread.n > 1)      bits.push(fmtTok(c.n) + ' tokens');
  if(spread.preset > 1) bits.push(SIZES[c.preset] || c.preset);
  if(spread.steps > 1)  bits.push(fmtN(c.steps) + ' steps');
  // Nothing we track varies -- which is normal for the causal baselines, where the difference
  // is the architecture and lives only in the name. Use the name, minus the corpus prefix it
  // shares with its neighbors, rather than printing the same label for every bar.
  if(!bits.length){
    const rest = c.base.startsWith(c.corpus + '_') ? c.base.slice(c.corpus.length + 1) : c.base;
    return rest.replace(/_/g, ' ') || (LANG_NAMES[c.corpus] || c.corpus);
  }
  return bits.join(' \u00b7 ');
}

const EXP_NAMES = {ladder:'Yoruba data ladder', multi:'Five languages',
                   eng:'English data ladder',
                   wikitext103:'WikiText-103 baselines',
                   batchtest:'Batch-size sweep', lrprobe:'Learning-rate sweep',
                   stabcheck:'Seed stability'};
const LANG_NAMES = {eng:'English', eng_1b:'English', swh:'Swahili', amh:'Amharic',
                    afr:'Afrikaans', som:'Somali', xho:'Xhosa', kin:'Kinyarwanda',
                    sna:'Shona', lug:'Luganda', wol:'Wolof', nya:'Chichewa', fra:'French', ind:'Indonesian', cmn:'Mandarin',
                    yor:'Yoruba', swh:'Swahili', hau:'Hausa', ibo:'Igbo'};

const MY_VERSION = '__PAGE_VERSION__';

async function tick(){
  try{
    const d = await (await fetch('/api/', {cache:'no-store'})).json();
    document.getElementById('stale').hidden =
      !d.page_version || d.page_version === MY_VERSION;

    // The cards are working and nothing on this page accounts for it. Say so rather than
    // rendering a confident zero -- this exact situation has been misread as "finished" four
    // times, and every previous fix made the declaration better instead of making the gap
    // visible.
    const un = document.getElementById('unexplained');
    un.hidden = !d.unexplained_work;
    if(d.unexplained_work){
      const cards = (d.busy_cards || [])
        .map(c => `card ${c.index} at ${c.util}%`).join(' and ');
      un.innerHTML = `<b>Work in progress that this page cannot account for.</b> ${cards}, `
        + `but no run is writing a curve and nothing in the queue is marked running. `
        + `Most likely a study is pretraining before it declared its cells, or is writing under `
        + `a tag the plan does not contain. The machine is busy — this panel is not.`;
    }

    renderQueue(d.fleet);
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
      + '<td class="num">batch</td><td class="num">took</td>'
      + '<td class="num">how it converged</td></tr>'
      + past.map(r=>`<tr>
      <td>${r.description||r.tag}${r.stalled?' <span class="pill warn">stalled</span>':''}
          <div class="tagsm">${r.tag}</div></td>
      <td class="num ${qcls(r.quality)}">${r.quality==null?'–':Math.round(r.quality*100)+'%'}</td>
      <td class="num">${r.val_loss.toFixed(3)}</td>
      <td class="num">${fmtN(Math.exp(r.val_loss))} words</td>
      <td class="num">${fmtN(r.steps_total)}</td>
      <td class="num">${r.batch||'–'}</td>
      <td class="num">${fmtT(r.elapsed)}</td>
      <td class="mini-cell">${miniChart(r.curve, r.random_loss)}</td></tr>`).join('');

    // "nothing training right now" was wrong far more often than it was right: it counted only
    // runs writing a curve, so a study fine-tuning for twenty minutes read as an idle machine.
    // Studies get named here, because a study process IS work in progress.
    const act = (d.fleet && d.fleet.groups || []).filter(g => g.active);
    const where = g => {
      // A sharded study is several workers, possibly on different cards. Naming one of them
      // would be the same class of wrong as the panel that named one card while two were busy.
      const w = g.workers || [];
      const cards = [...new Set(w.map(x => x.gpu).filter(x => x != null))];
      if (w.length > 1) return ` · ${w.length} workers${cards.length ? ' on card ' + cards.join(' and ') : ''}`;
      return g.gpu != null ? ` · card ${g.gpu}` : '';
    };
    document.getElementById('sub').textContent =
      live.length ? `${live.length} training`
      : act.length ? act.map(g => g.study + where(g)).join(' · ')
      : (d.busy_cards || []).length ? 'a card is busy — see below'
      : 'nothing training right now';
    // Show when the SERVER started. An edited file does not reach a process that is already
    // running, and a dashboard that silently serves stale code wastes a lot of confusion.
    const up = new Date(d.server_started*1000).toLocaleTimeString();
    document.getElementById('clock').textContent =
      `${new Date().toLocaleTimeString()} · server up since ${up}`;
  }catch(e){ document.getElementById('sub').textContent = 'lost connection to the server'; }
}
['axis','measure','experiment'].forEach(id =>
  document.getElementById(id).addEventListener('change', tick));
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
