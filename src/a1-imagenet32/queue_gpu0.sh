#!/bin/bash
P="${PYTHON_BIN:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
# The big ViT is the long pole (~5h). Give it a card to itself.
"$P" -W ignore train.py --model vit_base --gpu 0 --epochs 40 --batch 512 > logs/vit_base.log 2>&1
