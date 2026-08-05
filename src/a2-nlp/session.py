"""Session bootstrap for notebooks running on a Colab runtime.

Why this exists
---------------
A Colab runtime's *disk* is shared between notebooks; its *Python process* is not. Two
notebooks on the same runtime both see /content/WashingtonCsed504 and the prepared corpus,
but each has its own kernel, so neither inherits the other's sys.path, working directory,
imports or environment variables.

That means every notebook needs the same setup, but only the first one on a given runtime
pays for the expensive parts. Everything here is idempotent: the clone, the pip install and
the corpus preparation are all skipped when they have already been done.

Usage — put this stub at the top of any new notebook and nothing else:

    import os, sys
    REPO = '/content/WashingtonCsed504'
    if not os.path.exists(REPO):
        !git clone -q https://github.com/patlkwok/WashingtonCsed504.git {REPO}
    sys.path.insert(0, f'{REPO}/src/a2-nlp')
    import session; factory = session.start()

Options:
    session.start(corpus='yor')     # prepare a different corpus
    session.start(prepare=False)    # fine-tuning only, no pretraining corpus needed
    session.start(pull=False)       # skip git pull (faster, but you may be running stale code)

This is a local working file. It is committed to the fork so that the runtime's clone
includes it, but it is not part of the upstream project and should not be sent there.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

REPO_DIR = "/content/WashingtonCsed504"
FACTORY_DIR = f"{REPO_DIR}/src/a2-nlp"

# (import name, pip name). fasttext is deliberately absent: it is needed only by GlotLID for
# the one-off language-ID check, and the current build fails on Python 3.12, where `imp` was
# removed. Install it in that notebook alone.
PACKAGES = [
    ("datasets", "datasets"),
    ("transformers", "transformers"),
    ("tokenizers", "tokenizers"),
    ("seqeval", "seqeval"),
]

TOKENIZERS = {"yor": ("yor_Latn", "yo", "tokenizers/yor-bpe16k", "15abd33de5af")}


def _sh(cmd: str) -> tuple[int, str]:
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def _missing_packages() -> list[str]:
    out = []
    for mod, pip_name in PACKAGES:
        try:
            importlib.import_module(mod)
        except ImportError:
            out.append(pip_name)
    return out


def gpu_report() -> dict:
    """Print and return what card this runtime has. Cheap; call it before anything slow."""
    import torch

    info = {"python": sys.version.split()[0], "cuda": torch.cuda.is_available()}
    if not info["cuda"]:
        print("python", info["python"], "| NO GPU — Runtime > Change runtime type")
        return info

    p = torch.cuda.get_device_properties(0)
    info.update(
        name=p.name,
        vram_gb=round(p.total_memory / 1e9),
        sm=f"sm_{p.major}{p.minor}",
        bf16=torch.cuda.is_bf16_supported(),
    )
    print(
        f"python {info['python']} | {info['name']} "
        f"({info['vram_gb']} GB, {info['sm']}) | bf16: {info['bf16']}"
    )
    if not info["bf16"]:
        print(
            "  WARNING: no bf16 (likely a T4). Training falls back to fp16 + GradScaler,\n"
            "  the path the 86M pretraining runs already collapsed in. Fine-tuning is fine;\n"
            "  pretraining on this card is not."
        )
    return info


def start(corpus: str = "yor", pull: bool = True, install: bool = True,
          prepare: bool = True, quiet: bool = False):
    """Make this kernel ready to use the factory. Safe to call from any notebook.

    Returns the mlm_api module.
    """
    log = (lambda *a: None) if quiet else print

    info = gpu_report()

    # --- repo on the runtime's disk (shared between notebooks) ---------------
    if not os.path.isdir(REPO_DIR):
        raise RuntimeError(
            f"{REPO_DIR} not found. Clone it first — see this module's docstring.\n"
            "Clone YOUR FORK, not upstream, or the runtime gets a repo without your work."
        )
    if pull:
        rc, out = _sh(f"cd {REPO_DIR} && git pull -q")
        if rc:
            log(f"  git pull failed (continuing with the existing clone): {out[:200]}")

    # --- packages (skipped entirely if already importable) -------------------
    if install:
        missing = _missing_packages()
        if missing:
            log(f"  installing {', '.join(missing)} ...")
            _sh(f"{sys.executable} -m pip install -q -U {' '.join(missing)}")
        else:
            log("  packages already present")

    # --- in-process wiring (NOT shared between notebooks) --------------------
    if FACTORY_DIR not in sys.path:
        sys.path.insert(0, FACTORY_DIR)
    os.chdir(FACTORY_DIR)

    import mlm_api as factory

    # --- corpus (on disk, so shared; prepare_corpus no-ops if already done) --
    if prepare:
        if corpus not in TOKENIZERS:
            raise ValueError(f"unknown corpus {corpus!r}; known: {list(TOKENIZERS)}")
        lang, wiki, tok_path, fingerprint = TOKENIZERS[corpus]
        factory.prepare_corpus(corpus, lang=lang, wiki=wiki, tokenizer=tok_path)
        got = factory.corpus_info(corpus)["tokenizer_fingerprint"]
        if got != fingerprint:
            raise RuntimeError(
                f"tokenizer fingerprint {got} != expected {fingerprint}.\n"
                "A different vocabulary makes every loss incomparable with the recorded "
                "results. Do not proceed."
            )
        log(f"  corpus '{corpus}' ready, tokenizer {got}")

    log(f"ready — cwd {os.getcwd()}")
    return factory


def mount_drive(require: bool = True) -> str:
    """Mount Drive and verify it. Only needed when saving something unregenerable.

    `mkdir -p /content/drive/...` SUCCEEDS when Drive is unmounted — it silently creates an
    ordinary local directory, the copy reports success, and everything dies with the session.
    That is the easiest way to lose results while believing they are backed up.
    """
    from google.colab import drive

    drive.mount("/content/drive")
    ok = os.path.ismount("/content/drive")
    if require and not ok:
        raise RuntimeError("Drive is NOT mounted — do not write under /content/drive")
    return "/content/drive/MyDrive"


def save_results(dest: str = "/content/drive/MyDrive/csed504-runs") -> int:
    """Copy run records off the runtime's disposable disk. Call at the end of a session."""
    mount_drive(require=True)
    os.makedirs(dest, exist_ok=True)
    runs = os.path.join(FACTORY_DIR, "runs")
    n = 0
    for f in os.listdir(runs):
        if f.endswith((".json", ".jsonl")):
            _sh(f"cp '{os.path.join(runs, f)}' '{dest}/'")
            n += 1
    print(f"copied {n} result files -> {dest}")
    return n
