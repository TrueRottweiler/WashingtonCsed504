"""Shared train/eval loop for the CIFAR notebooks — device-agnostic, so CPU and GPU runs are the
same code path.  The only things that change between backends are passed in: `channels_last` and
`amp_dtype` (both come from `cifar_pipeline.make_loaders`'s cfg).

Recipe is the part-1 default: SGD (nesterov) or AdamW, 5-epoch linear warmup -> cosine decay, label
smoothing, gradient clipping, AMP autocast, optional mixup.  A GradScaler is used only for fp16
(cuda); bf16 (the default on both Zen4 CPU and Blackwell GPU) needs none.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn


def _prep(x, channels_last):
    return x.contiguous(memory_format=torch.channels_last) if channels_last else x


def _mixup(x, y, alpha=0.2):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam


@torch.no_grad()
def evaluate(model, test_iter, device, *, channels_last=False, amp_dtype=None):
    model.eval()
    c1 = c5 = n = 0
    for x, y in test_iter:
        x, y = _prep(x.to(device, non_blocking=True), channels_last), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(x)
        _, pred = out.float().topk(5, dim=1)
        hits = pred.eq(y.view(-1, 1))
        c1 += hits[:, :1].any(1).sum().item()
        c5 += hits.any(1).sum().item()
        n += y.size(0)
    model.train()
    return c1 / n, c5 / n


def train(model, train_iter, test_iter, device, *, epochs, channels_last=False, amp_dtype=None,
          opt='sgd', lr=0.1, wd=5e-4, momentum=0.9, warmup=5, label_smoothing=0.1, clip=1.0,
          mixup=False, log=print):
    """Train `model` and return (history, best_top1). Works identically on CPU and CUDA."""
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model = model.to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    fused = device.type == 'cuda'
    optimizer = (torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, fused=fused)
                 if opt == 'adamw' else
                 torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True,
                                 weight_decay=wd, fused=fused))
    warm = torch.optim.lr_scheduler.LinearLR(optimizer, 0.01, total_iters=warmup)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], [warmup])

    use_scaler = amp_dtype is torch.float16 and device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    history = {'train_loss': [], 'top1': [], 'top5': [], 'img_s': []}
    best = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, seen = time.time(), 0
        loss_sum = torch.zeros((), device=device)
        for x, y in train_iter:
            x = _prep(x.to(device, non_blocking=True), channels_last)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                if mixup:
                    xm, ya, yb, lam = _mixup(x, y)
                    out = model(xm)
                    loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
                else:
                    loss = criterion(model(x), y)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            loss_sum += loss.detach() * y.size(0)
            seen += y.size(0)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.time() - t0
        train_loss = (loss_sum / seen).item()
        if train_loss != train_loss:
            raise RuntimeError('loss is NaN — the run has diverged (lower the LR)')
        scheduler.step()

        top1, top5 = evaluate(model, test_iter, device, channels_last=channels_last, amp_dtype=amp_dtype)
        best = max(best, top1)
        history['train_loss'].append(train_loss)
        history['top1'].append(top1)
        history['top5'].append(top5)
        history['img_s'].append(seen / dt)
        log(f'epoch {epoch:3d}/{epochs}  loss {train_loss:.3f}  |  '
            f'val top1 {top1:6.2%} top5 {top5:6.2%}  |  {dt:5.1f}s  {seen/dt:8,.0f} img/s')
    return history, best
