"""
models.py: the two language-model families (LSTM + GPT), sized so the fair fight is fair.

This is the a2 counterpart of a1-cv/models.py, duplicated rather than shared on purpose: the
factory's seam is "a builders dict the trainer looks names up in", not a common base class.

The matching discipline carries over from Part 1, with one text-specific wrinkle. In Part 1 we
parameter-matched resnet18 (11.7M) against vit (11.0M) so neither could win by being larger. In
NLP the embedding table complicates that: with a 16k BPE vocabulary the tied token embedding is
~6M parameters, and BOTH models carry an identical copy of it (same vocab, same width, tied to
the output head the same way). So the honest match is on the *backbone* -- the recurrent stack
against the transformer stack -- with the embedding held as a shared constant between them.
n_backbone_params() reports that number, and the trainer prints both.

The four builders and their jobs, mirroring the Part 1 table:

    lstm        recurrent    ~10.6M backbone   the recurrence baseline        (= resnet18)
    gpt         transformer  ~10.7M backbone   the parameter-matched fight    (= vit)
    lstm_large  recurrent    ~2x               "would a bigger LSTM win?"     (= resnet50)
    gpt_medium  transformer  ~3.5x             "does scaling the gpt help?"   (= vit_base)

gpt deliberately reuses the Part 1 ViT's dims (d=384, 6 layers, 6 heads): same backbone budget,
different modality. The attention itself uses F.scaled_dot_product_attention rather than the
hand-written version in hello_text.ipynb -- the notebook exists to show the mechanics, the
factory exists to train fast, and SDPA dispatches to a fused kernel that never materializes the
T x T score matrix. That fusion is the win worth having: the unfused math path measures 7.5x
slower on a training-shaped batch.

Which fused kernel you get varies by platform, so it is worth knowing what actually runs. The
Windows wheels are built without the bundled FlashAttention -- can_use_flash_attention is False
here whatever the enable flags say -- so this box runs the CUTLASS memory-efficient kernel.
cuDNN ships its own flash kernel, it is available, and it measured 1.14x on attention alone but
only 1.7% on a full step at seq_len 256, so the default stands rather than being pinned.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LstmLM(nn.Module):
    """Embedding -> multi-layer cuDNN LSTM -> projection back to embed width -> tied softmax head.

    The projection (hidden -> embed_dim) exists so the head can share the embedding matrix even
    though the LSTM's hidden width differs from the embedding width. Weight tying is the same
    Press & Wolf trick hello_text.ipynb uses, and both families get it, so neither gains a free
    ~6M-parameter advantage from an untied head.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 384, hidden: int = 864,
                 layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.drop = nn.Dropout(dropout)

        # The stack is single-layer nn.LSTM modules with our own nn.Dropout between them, NOT
        # one multi-layer nn.LSTM with dropout=. The semantics are identical (cuDNN's inter-layer
        # dropout is exactly "dropout on each layer's output except the last") and each layer
        # still runs the fused cuDNN kernel -- but a multi-layer LSTM with dropout>0 allocates a
        # cuDNN dropout state whose destructor fail-fast crashes (0xC0000409) at interpreter
        # shutdown on Windows, turning every clean run into a nonzero exit code the fleet would
        # report as FAILED. Stacking sidesteps the state entirely.
        # batch_first so the (B, T) -> (B, T, D) flow reads the same as the GPT below.
        self.layers = nn.ModuleList([
            nn.LSTM(embed_dim if i == 0 else hidden, hidden, batch_first=True)
            for i in range(layers)])
        self.proj = nn.Linear(hidden, embed_dim, bias=False)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.tok_embed.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.tok_embed(idx))           # (B, T, D)
        for i, layer in enumerate(self.layers):
            x, _ = layer(x)                          # (B, T, H)  fresh zero state each window
            if i < len(self.layers) - 1:
                x = self.drop(x)
        return self.head(self.proj(self.drop(x)))    # (B, T, V)


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention via F.scaled_dot_product_attention (Flash path when available).

    Functionally identical to the step-by-step version in hello_text.ipynb; see that notebook for
    the mechanics. Here the mask, softmax, and weighted sum happen inside one fused kernel.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
        self.num_heads = num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.num_heads, D // self.num_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, D))


class Block(nn.Module):
    """Pre-LayerNorm transformer block: x + attn(norm(x)), then x + mlp(norm(x))."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    """Decoder-only transformer LM: token + learned position embeddings, blocks, tied head."""

    def __init__(self, vocab_size: int, seq_len: int, embed_dim: int = 384,
                 num_heads: int = 6, num_layers: int = 6, ffn_dim: int = 1536,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(seq_len, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.tok_embed.weight

        # GPT-2-style init: N(0, 0.02) everywhere. Torch's default Linear init scales with fan-in,
        # which is fine for the LSTM but leaves a pre-LN transformer's residual stream too hot at
        # this depth; the flat 0.02 is what the whole GPT line trains with.
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.seq_len, f'sequence length {T} > context window {self.seq_len}'
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_embed(idx) + self.pos_embed(pos))
        x = self.blocks(x)
        return self.head(self.norm(x))


# The builders dict is the factory's seam: train_run.py looks names up here, exactly like
# a1-cv/models.py BUILDERS. Every builder takes (vocab_size, seq_len) so the same name works
# unchanged across the char-level and BPE rungs, whose vocabularies differ by 100x.

def make_lstm(vocab_size: int, seq_len: int) -> nn.Module:
    # hidden=864 x 2 layers lands the backbone at ~10.6M, matching make_gpt below.
    return LstmLM(vocab_size, embed_dim=384, hidden=864, layers=2, dropout=0.3)


def make_lstm_large(vocab_size: int, seq_len: int) -> nn.Module:
    # The capacity control: ~2x the backbone. Answers "was the baseline LSTM just too small?"
    return LstmLM(vocab_size, embed_dim=384, hidden=1024, layers=3, dropout=0.3)


def make_gpt(vocab_size: int, seq_len: int) -> nn.Module:
    # d384/L6/H6 -- the Part 1 ViT's dims, reused on purpose (see the module docstring).
    return GPT(vocab_size, seq_len, embed_dim=384, num_heads=6, num_layers=6,
               ffn_dim=1536, dropout=0.1)


def make_gpt_medium(vocab_size: int, seq_len: int) -> nn.Module:
    # The other capacity control: ~3.5x the backbone.
    return GPT(vocab_size, seq_len, embed_dim=512, num_heads=8, num_layers=12,
               ffn_dim=2048, dropout=0.1)


BUILDERS = {
    'lstm': make_lstm,
    'gpt': make_gpt,
    'lstm_large': make_lstm_large,
    'gpt_medium': make_gpt_medium,
}


def build(name: str, vocab_size: int, seq_len: int) -> nn.Module:
    return BUILDERS[name](vocab_size, seq_len)


def n_params(model: nn.Module) -> int:
    """Unique trainable parameters. Weight tying means head and tok_embed share one tensor, so we
    de-dup by id() -- counting the shared matrix twice would overstate both models equally, but
    the printed number should still be the true one."""
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) not in seen and p.requires_grad:
            seen.add(id(p))
            total += p.numel()
    return total


def n_backbone_params(model: nn.Module) -> int:
    """Parameters excluding the token and position embeddings -- the number the two families are
    matched on. The tied head is the same tensor as tok_embed, so excluding tok_embed excludes it."""
    skip = {id(model.tok_embed.weight)}
    if hasattr(model, 'pos_embed'):
        skip.add(id(model.pos_embed.weight))
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) not in seen and id(p) not in skip and p.requires_grad:
            seen.add(id(p))
            total += p.numel()
    return total
