"""Derive what each module actually depends on, so the architecture map is measured not guessed.

Reports three things per file: which sibling modules it imports, which third-party frameworks it
touches, and whether it mentions anything task-specific (masking, tokenizers, images). That last
column is what separates "generic infrastructure" from "this problem only".
"""
import ast
import os
import re
from collections import defaultdict

ROOTS = ['src/common', 'src/a1-cv', 'src/a2-nlp']
LOCAL = set()
for root in ROOTS:
    for f in os.listdir(root):
        if f.endswith('.py'):
            LOCAL.add(f[:-3])

# Words that betray a commitment to one problem. Counted, not judged -- a file with none of these
# is a candidate for sharing; a file with many is specific by construction.
MARKERS = {
    'masked-LM': r'\bmlm|\bmask_id|masked_lm|mlm_prob',
    'tokenizer': r'tokenizer|\bbpe\b|vocab_size|AutoTokenizer',
    'text': r'seq_len|token|corpus|perplexity',
    'image': r'\bimage|cifar|imagenet|\bpixel|top-?1|RandomCrop',
    'torch': r'\btorch\b',
    'hf': r'transformers|datasets|huggingface',
}

rows = []
for root in ROOTS:
    for f in sorted(os.listdir(root)):
        if not f.endswith('.py'):
            continue
        path = os.path.join(root, f)
        src = open(path, encoding='utf-8', errors='replace').read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        imports = set()
        third = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    top = a.name.split('.')[0]
                    (imports if top in LOCAL else third).add(top)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                top = node.module.split('.')[0]
                (imports if top in LOCAL else third).add(top)
        hits = {k: len(re.findall(v, src, re.I)) for k, v in MARKERS.items()}
        rows.append({'dir': root.split('/')[-1], 'file': f, 'loc': src.count('\n') + 1,
                     'imports': sorted(imports),
                     'third': sorted(t for t in third if t in
                                     ('torch', 'transformers', 'datasets', 'tokenizers',
                                      'numpy', 'pandas', 'matplotlib', 'seqeval')),
                     'hits': hits})

print(f"{'dir':8s} {'file':22s} {'loc':>5s}  {'imports':34s} {'markers'}")
print('-' * 118)
for r in rows:
    marks = ' '.join(f'{k}:{v}' for k, v in r['hits'].items() if v)
    print(f"{r['dir']:8s} {r['file']:22s} {r['loc']:5d}  {','.join(r['imports'])[:33]:34s} {marks[:52]}")

print('\n\n=== who imports what (reverse edges) ===')
rev = defaultdict(set)
for r in rows:
    for i in r['imports']:
        rev[i].add(f"{r['dir']}/{r['file']}")
for mod in sorted(rev):
    print(f'  {mod:18s} <- {", ".join(sorted(rev[mod]))}')

print('\n\n=== files importing NOTHING local (leaves / self-contained) ===')
for r in rows:
    if not r['imports']:
        print(f"  {r['dir']:8s} {r['file']}")
