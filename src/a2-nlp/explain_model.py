"""
explain_model.py -- ask a trained checkpoint to fill in blanks, and see what it knows.

A validation loss of 2.886 is not an intuition. This turns it into one, two ways:

  fill-in-the-blank   hide a word in real held-out text and print the model's top guesses next
                      to an untrained model's, with the rank of the true answer in each. The
                      untrained model buries the answer around position 12,000 of 16,000; a
                      trained one proposes real Yoruba function words that fit the slot.

  effective choices   exp(loss) is the number of words the model is still choosing between. It
                      is the same number as perplexity, but phrased so it can be said out loud:
                      training narrowed the field from 16,000 candidates to about 18.

Usage:
    python explain_model.py --corpus yor
    python explain_model.py --corpus yor --model runs/yor_2M_3k_s0 --n 6
    python explain_model.py --corpus yor --text "Àwọn ọmọ ilé ìwé lọ sí ọjà ní àárọ̀"
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch

import mlm_api as factory
import text_data as _text


def load(path, device):
    from transformers import AutoModelForMaskedLM
    return AutoModelForMaskedLM.from_pretrained(path).to(device).eval()


def held_out_sentences(corpus: str, tok, n: int, offset: int = 200) -> list[str]:
    """Sentences from the validation split -- text no training run ever saw.

    Resolved through text_data.DATA_DIR rather than off this file's directory, so it follows a
    corpus that has been redirected elsewhere -- which is what a Colab user does when caching
    prepared corpora on Drive so a session restart does not re-train the tokenizer.
    """
    path = os.path.join(_text.DATA_DIR, corpus, 'val_tokens.npy')
    val = np.load(path, mmap_mode='r')
    text = tok.decode(np.asarray(val[offset:offset + 120 * n]).tolist())
    return [s.strip() for s in text.split('.') if 60 < len(s.strip()) < 150][:n]


def compare_on(sentence: str, tok, models: dict, device, mask_id: int) -> None:
    ids = tok(sentence, return_tensors='pt', truncation=True, max_length=128)['input_ids']
    if ids.shape[1] < 12:
        return
    pos = ids.shape[1] // 2
    true_id = ids[0, pos].item()
    masked = ids.clone()
    masked[0, pos] = mask_id

    shown = tok.decode(masked[0].tolist()).replace(tok.mask_token, ' ___ ')
    print(f'\n  sentence : {" ".join(shown.split())[:112]}')
    print(f'  hidden   : {tok.decode([true_id])!r}')

    for name, model in models.items():
        with torch.no_grad():
            logits = model(input_ids=masked.to(device)).logits[0, pos]
        probs = logits.softmax(-1)
        top = probs.topk(5)
        guesses = ', '.join(f'{tok.decode([i]).strip()!r} {p:.0%}'
                            for p, i in zip(top.values.tolist(), top.indices.tolist()))
        # Rank of the true answer: how many tokens the model considered more likely.
        rank = int((probs > probs[true_id]).sum().item()) + 1
        print(f'  {name:<10}: {guesses}')
        print(f'  {"":<10}  true answer ranked #{rank:,} of {len(probs):,}')


def main():
    p = argparse.ArgumentParser(description='See what a pretrained checkpoint learned.')
    p.add_argument('--corpus', required=True)
    p.add_argument('--model', default=None,
                   help='checkpoint dir (default: the best completed run for this corpus)')
    p.add_argument('--control', default=None,
                   help='untrained comparison (default: runs/<corpus>_random_init)')
    p.add_argument('--text', default=None, help='use this sentence instead of held-out text')
    p.add_argument('--n', type=int, default=4, help='how many sentences to try')
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    tok = factory.load_tokenizer(args.corpus)

    model_path = args.model
    if model_path is None:
        runs = [r for r in factory.results(f'{args.corpus}_*') if 'val_loss' in r]
        if not runs:
            raise SystemExit(f'no completed runs for {args.corpus!r} -- train one first')
        best = min(runs, key=lambda r: r['val_loss'])
        model_path = best['path']
        print(f'using best run: {best["tag"]} (val loss {best["val_loss"]:.3f})')

    control_path = args.control or os.path.join(factory.RUNS, f'{args.corpus}_random_init')
    models = {'trained': load(model_path, device)}
    if os.path.exists(os.path.join(control_path, 'config.json')):
        models['untrained'] = load(control_path, device)
    else:
        print(f'(no control at {control_path} -- run: python mlm_run.py '
              f'--corpus {args.corpus} --random-init)')

    print('\n' + '=' * 78)
    print('WHAT THE MODEL PREDICTS WHEN A WORD IS HIDDEN')
    print('=' * 78)
    sentences = [args.text] if args.text else held_out_sentences(args.corpus, tok, args.n)
    for s in sentences:
        compare_on(s, tok, models, device, tok.mask_token_id)

    print('\n' + '=' * 78)
    print('WHAT THE LOSS NUMBERS MEAN')
    print('=' * 78)
    vocab = factory.corpus_info(args.corpus)['vocab_size']
    print(f'  exp(loss) is how many words the model is still choosing between, '
          f'out of {vocab:,}.\n')
    rows = [('untrained (predicts uniformly)', math.log(vocab))]
    rows += [(r['tag'], r['val_loss'])
             for r in sorted(factory.results(f'{args.corpus}_*'), key=lambda r: -r['val_loss'])]
    for label, loss in rows:
        print(f'  {label:<34} loss {loss:6.3f}  ->  {math.exp(loss):>8,.0f} words')


if __name__ == '__main__':
    main()
