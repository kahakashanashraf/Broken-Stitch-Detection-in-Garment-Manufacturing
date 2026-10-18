#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/run_matrix.sh /path/to/approved_split.csv /path/to/vgg16_weights.pth
# Before running: source_id parent lineage must be audited and exact leakage check must pass.
M="${1:?Pass audited split CSV}"; W="${2:?Pass official torchvision VGG-16 ImageNet state_dict}"
for FILTER in none gaussian bilateral median; do
  python -m broken_stitch.cli train --manifest "$M" --outdir "runs/cnn_filter_${FILTER}" --model cnn --filter "$FILTER" --epochs 30 --patience 6
 done
for ACT in relu softmax tanh sigmoid selu elu; do
  python -m broken_stitch.cli train --manifest "$M" --outdir "runs/cnn_activation_${ACT}" --model cnn --activation "$ACT" --epochs 30 --patience 6
 done
python -m broken_stitch.cli train --manifest "$M" --outdir runs/vgg_linear --model vgg_linear --vgg-weights "$W" --lr 0.0001 --epochs 30 --patience 6
python -m broken_stitch.cli train --manifest "$M" --outdir runs/vgg_finetune --model vgg_finetune --vgg-weights "$W" --lr 0.00001 --epochs 30 --patience 6
for SEED in 13 29 47; do
  python -m broken_stitch.cli train --manifest "$M" --outdir "runs/vgg_linear_seed_${SEED}" --model vgg_linear --vgg-weights "$W" --seed "$SEED" --lr 0.0001 --epochs 30 --patience 6
 done
# Before any held-out test evaluation, choose the final model using VALIDATION ONLY.
# Then run the evaluator once:
# python -m broken_stitch.cli evaluate --manifest "$M" --checkpoint runs/vgg_finetune/best.pt --split test --out final_test.json
# Ensemble of three independently trained frozen-VGG heads:
# python -m broken_stitch.cli ensemble --manifest "$M" --checkpoints runs/vgg_linear_seed_{13,29,47}/best.pt --split val --out ensemble_val.json