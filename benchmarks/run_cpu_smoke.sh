#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTHONPATH=${PYTHONPATH:-src}

rm -rf results/smoke-resnet results/smoke-vit
python src/a1-cv/search_cnn.py \
  --config configs/searches/smoke_resnet.toml --skip-calibration
python src/a1-cv/search_transformer.py \
  --config configs/searches/smoke_vit.toml --skip-calibration
