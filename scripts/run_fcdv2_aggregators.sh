#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# FCDv2 Aggregator Comparison — VAE / GMM / NF (RealNVP)
#
# Runs the three pluggable style aggregators on PACS sequentially:
#   1. GMM      (k-component Gaussian mixture over z_env)
#   2. VAE      (amortised latent posterior over z_env)
#   3. RealNVP  (normalising flow over z_env)
#
# Usage:
#   bash scripts/run_fcdv2_aggregators.sh                       # resnet18, seed=1001
#   bash scripts/run_fcdv2_aggregators.sh 1002                  # custom seed
#   bash scripts/run_fcdv2_aggregators.sh 1001 resnet50         # custom seed + backbone
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

SEED="${1:-1001}"
BACKBONE="${2:-resnet50}"

case "${BACKBONE}" in
  resnet18) PREFIX="configs/fcdv2/fcdv2_pacs"     ;;
  resnet50) PREFIX="configs/fcdv2/fcdv2_pacs_r50" ;;
  *)
    echo "error: unsupported backbone '${BACKBONE}' (expected resnet18 or resnet50)" >&2
    exit 2
    ;;
esac

CONFIGS=(
  "${PREFIX}_vae.json"
  "${PREFIX}_gmm.json"
  "${PREFIX}_realnvp.json"
)

for cfg in "${CONFIGS[@]}"; do
  echo "═══════════════════════════════════════════════════════════"
  echo "  Running: ${cfg} (seed=${SEED})"
  echo "═══════════════════════════════════════════════════════════"
  python main.py --config_file "${cfg}" --seed "${SEED}"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All three FCDv2 aggregator runs complete (${BACKBONE}, seed=${SEED})."
echo "═══════════════════════════════════════════════════════════"
