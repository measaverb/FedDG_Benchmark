#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Augmentation Source Comparison — FCD Ablation (CelebA R50)
#
# Runs all four augmentation conditions sequentially:
#   1. Isotropic Gaussian N(0, I)
#   2. Local empirical Gaussian N(μ_i, Σ_i)
#   3. Naive unimodal global Gaussian N(μ_global, Σ_global)
#   4. Proposed multi-modal Global GMM (control)
#
# Usage:
#   bash scripts/run_aug_ablation.sh          # default seed 1001
#   bash scripts/run_aug_ablation.sh 1002     # custom seed
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

SEED="${1:-1001}"

CONFIGS=(
  "configs/fcd/fcd_celeba_r50_aug_isotropic.json"
  "configs/fcd/fcd_celeba_r50_aug_local.json"
  "configs/fcd/fcd_celeba_r50_aug_unimodal.json"
  # "configs/fcd/fcd_celeba_r50_aug_gmm.json"
)

for cfg in "${CONFIGS[@]}"; do
  echo "═══════════════════════════════════════════════════════════"
  echo "  Running: ${cfg} (seed=${SEED})"
  echo "═══════════════════════════════════════════════════════════"
  python main.py --config_file "${cfg}" --seed "${SEED}"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All four augmentation ablation runs complete."
echo "═══════════════════════════════════════════════════════════"
