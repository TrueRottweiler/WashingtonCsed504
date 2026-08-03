"""
Per-card trainer for an LSTM or GPT language model on shakespeare, wikitext2, or wikitext103.

Duplicated from a1-cv/train_run.py; the shape is unchanged -- one process, one GPU, one model,
checkpoint every epoch, --resume, GPU-resident data, JSONL history -- and the deltas are the
NLP ones: datasets are token streams, the metric is perplexity (lower is better, so is_best
flips), batches are (sequences x seq_len) windows, and the family split is lstm-vs-gpt instead
of resnet-vs-vit.

Usage and examples:
    python train_run.py --model gpt --dataset wikitext103 --gpu 0
    python train_run.py --model lstm --dataset wikitext2 --epochs 30 --gpu 1
    python train_run.py --model gpt --dataset shakespeare --smoke-test    # wiring check, exits
    python train_run.py --model lstm --dataset wikitext103 --resume

Recipes (the defaults wired up below; both families use AdamW):
    lstm: AdamW, lr 1e-3, no weight decay -- the dropout in the model is its regularizer.
    gpt:  AdamW, lr 3e-4, weight decay 0.1, betas (0.9, 0.95) -- the standard GPT recipe.

Both get the warmup (2 epochs) and cosine decay, and both get grad clipping at 1.0. That is a
deliberate difference from Part 1, where clipping was per-family because it cost ResNet-18 five
points: for language models clipping is standard practice in BOTH families' canonical recipes
(a bad window can spike the loss for either), so applying it to both cannot tilt the race.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn as nn

import models as M
import text_data as T
import train_loop as E

OUT_DIR = os.path.join(os.path.dirname(__file__), 'runs')

DATASETS = ['shakespeare', 'wikitext2', 'wikitext103']


def build_optimizer(model, name, batch_tokens, args):
    # Pick the recipe by model family, keyed with startswith for the same reason as a1: an exact
    # name test once handed resnet50 the wrong optimizer with no error. lstm* is one family,
    # gpt* the other, and a new size variant inherits its family's recipe automatically.
    #
    # The LR scales linearly with tokens-per-step from a 32k-token baseline (batch 128 x seq 256),
    # the LM analog of a1's batch/256 rule: change --batch or --seq-len and the step size follows.
    scale = batch_tokens / (128 * 256)
    if name.startswith('lstm'):
        lr = args.lr if args.lr else 1e-3 * scale
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, fused=True)
    else:
        lr = args.lr if args.lr else 3e-4 * scale
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1,
                                betas=(0.9, 0.95), fused=True)
    return opt, lr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=list(M.BUILDERS), required=True)
    p.add_argument('--dataset', choices=DATASETS, default='wikitext2',
                   help='which token stream to train on (default: wikitext2)')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=128, help='sequences per step')
    p.add_argument('--seq-len', type=int, default=256, help='context window in tokens')
    p.add_argument('--lr', type=float, default=None, help='override the scaled default')
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke-test', action='store_true', help='tiny subset, 2 epochs, then exit')
    p.add_argument('--tag', default=None, help='name for this run (default: dataset_model)')
    p.add_argument('--seed', type=int, default=42,
                   help='RNG seed (default 42); vary it to repeat a run and measure the spread')
    p.add_argument('--clip', type=float, default=1.0,
                   help='grad-norm clip for both families, 0 disables')
    p.add_argument('--store-dtype', choices=['auto', 'int16', 'int32'], default='auto',
                   help='width of the GPU-resident token store (default: narrowest that fits)')
    args = p.parse_args()

    # Name the run: <dataset>_<model>, with the smoke prefix applied last so it cannot be
    # escaped by an explicit --tag -- the exact lesson a1 learned when a smoke test once wrote
    # its throwaway epochs into a real run's history.
    base = f'{args.dataset}_{args.model}'
    tag = args.tag or base
    if args.smoke_test:
        tag = f'smoke-{tag}'

    # Pin this process to its card and switch on the fast-math paths, same as a1. seed keeps
    # runs comparable; comparable is not identical (cuDNN autotune), which is what --seed
    # repeats are for.
    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    # bf16 where the hardware has it (no GradScaler to babysit), fp16 plus scaler elsewhere.
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, f'{tag}.pt')
    jsonl_path = os.path.join(OUT_DIR, f'{tag}.jsonl')

    # A smoke test collapses to 2 epochs on a token-count subset so the whole thing finishes in
    # well under a minute. 500k train tokens is enough for the loss to visibly move.
    epochs = 2 if args.smoke_test else args.epochs
    subset = 500_000 if args.smoke_test else None
    val_subset = 100_000 if args.smoke_test else None

    # Load both splits straight onto the GPU. The footprints are tiny by a1 standards (WikiText-103
    # is ~0.23 GB resident), so the interesting print is the token count, not the gigabytes.
    print(f'[{tag}] device {device} ({torch.cuda.get_device_name(device)})')
    t_load = time.time()
    train_ds = T.GpuTokens(device, args.dataset, 'train', seq_len=args.seq_len, subset=subset,
                           store_dtype=args.store_dtype)
    val_ds = T.GpuTokens(device, args.dataset, 'val', seq_len=args.seq_len, subset=val_subset,
                         store_dtype=args.store_dtype)
    store = str(train_ds.store_dtype).replace('torch.', '')
    print(f'[{tag}] {args.dataset} resident on GPU: {train_ds.n:,} train + {val_ds.n:,} val tokens '
          f'({train_ds.gb() + val_ds.gb():.2f} GB as {store}, vocab {train_ds.vocab_size:,} '
          f'{train_ds.tokenizer}) in {time.time()-t_load:.0f}s')

    # Build the model on-device. We print total AND backbone parameters: total is what the card
    # holds, backbone is what the two families are matched on (see models.py on why they differ).
    model = M.build(args.model, train_ds.vocab_size, args.seq_len).to(device)
    print(f'[{tag}] {args.model}: {M.n_params(model):,} parameters '
          f'({M.n_backbone_params(model):,} backbone)')

    batch_tokens = args.batch * args.seq_len
    optimizer, lr = build_optimizer(model, args.model, batch_tokens, args)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype is torch.float16)

    # Linear warmup into cosine decay, identical to a1. A transformer's warmup is not optional
    # (the attention softmax saturates on early noisy batches without it), so both families get
    # it and we stop worrying about which one needed it.
    warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=args.warmup)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - args.warmup))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cos],
                                                      milestones=[args.warmup])

    # best is a perplexity now, so it starts at infinity and improves DOWNWARD; every comparison
    # below is flipped relative to a1's accuracy. Fresh runs never append to an old history.
    start_epoch, best, history = 1, float('inf'), []
    if not args.resume and os.path.exists(jsonl_path):
        os.remove(jsonl_path)
    if args.resume and os.path.exists(ckpt_path):
        e, best, history = E.load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, device)
        start_epoch = e + 1
        print(f'[{tag}] resumed from epoch {e} (best val ppl {best:.2f})')

    # Echo the resolved config so the log header records exactly what this run did. The dashboard
    # parses the "dataset <name> |" anchor out of this line, so keep its shape if you edit it.
    print(f'[{tag}] dataset {args.dataset} | seq_len {args.seq_len} | grad clip {args.clip} | '
          f'amp {str(amp_dtype).replace("torch.", "")}')
    print(f'[{tag}] {epochs} epochs, batch {args.batch} x {args.seq_len} tok '
          f'({batch_tokens:,} tok/step), lr {lr:.5f} ({args.warmup}-epoch warmup then cosine)')
    print(f'[{tag}] {train_ds.n_batches(args.batch):,} batches/epoch\n', flush=True)

    # The epoch loop: train, evaluate, step the schedule, log, checkpoint. Identical sequencing
    # to a1; only the "which direction is better" flipped.
    t_start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        tr = E.train_one_epoch(model, train_ds, optimizer, criterion, scaler, device,
                               args.batch, epoch, epochs,
                               amp_dtype=amp_dtype, clip=(args.clip or None))
        va = E.evaluate(model, val_ds, criterion, device, batch_size=args.batch,
                        amp_dtype=amp_dtype)
        scheduler.step()

        is_best = va['ppl'] < best
        best = min(best, va['ppl'])
        history.append({'epoch': epoch, 'train': tr, 'val': va})
        E.log_epoch(tag, epoch, epochs, tr, va, scheduler.get_last_lr()[0],
                    time.time() - t_start, device, is_best, jsonl_path)
        E.save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, epoch, best, history)

    total = time.time() - t_start
    print(f'\n[{tag}] DONE. best val ppl {best:.2f} in {E.fmt_time(total)}')
    json.dump({'tag': tag, 'dataset': args.dataset, 'model': args.model,
               'params': M.n_params(model), 'backbone_params': M.n_backbone_params(model),
               'epochs': epochs, 'batch': args.batch, 'seq_len': args.seq_len, 'lr': lr,
               'store_dtype': store, 'store_gb': train_ds.gb() + val_ds.gb(),
               'best_ppl': best, 'seconds': total, 'history': history},
              open(os.path.join(OUT_DIR, f'{tag}_result.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
