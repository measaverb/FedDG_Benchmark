#!/usr/bin/env bash
#
# Run a set of FedDG baselines on miniDomainNet (ResNet-50 backbone, the default).
#
# Each baseline reuses an existing PACS template config, patched at runtime to
# point at miniDomainNet via split_scheme="minidomainnet". Matches the PACS
# convention used throughout the repo: num_clients=10, iid=1.0, num_rounds=50
# (ERM uses 80, mirroring configs/erm/fedavg_erm_pacs.json).
#
# Default L2DO baked into minidomainnet.csv: val=painting, test=real.
# Train pool = {clipart, sketch}.
#
# Usage:
#   bash scripts/run_fdg_baselines_minidomainnet.sh                 # all baselines, default rounds, wandb on
#   ROUNDS=30 bash scripts/run_fdg_baselines_minidomainnet.sh       # override num_rounds for all
#   METHODS="ERM GroupDRO" bash scripts/run_fdg_baselines_minidomainnet.sh  # subset
#   NO_WANDB=1 bash scripts/run_fdg_baselines_minidomainnet.sh      # disable wandb (smoke/debug)
#   WANDB_GROUP=my_sweep bash scripts/run_fdg_baselines_minidomainnet.sh  # override default group
#
# Logs land in logs/fdg_minidomainnet/<server>_<client>.{log,json}.
# Continues to the next baseline on per-method failure.

set -u

WANDB_FLAG=""
if [[ "${NO_WANDB:-0}" != "0" ]]; then
  WANDB_FLAG="--no_wandb"
fi

# All methods in a sweep land in the same wandb group/project for easy comparison.
WANDB_GROUP="${WANDB_GROUP:-minidomainnet_n10_iid1}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-FedDG_Benchmark_minidomainnet}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="logs/fdg_minidomainnet"
CONFIG_DIR="${LOG_DIR}/configs"
mkdir -p "$LOG_DIR" "$CONFIG_DIR"

# (client_method, server_method, template_config, override_rounds)
# override_rounds left blank = inherit from template (ERM=80, others=50)
BASELINES=(
  "ERM      FedAvg  configs/erm/fedavg_erm_pacs.json"
  "GroupDRO FedAvg  configs/groupdro/fedavg_groupdro_pacs.json"
  "IRM      FedAvg  configs/irm/fedavg_irm_pacs.json"
  "Coral    FedAvg  configs/coral/fedavg_coral_pacs.json"
  "MMD      FedAvg  configs/mmd/fedavg_mmd_pacs.json"
  "VREx     FedAvg  configs/vrex/fedavg_vrex_pacs.json"
  "FedProx  FedAvg  configs/fedprox/fedavg_fedprox_pacs.json"
)

# Optional method filter (e.g. METHODS="ERM IRM")
filter="${METHODS:-}"

declare -A RESULTS

for entry in "${BASELINES[@]}"; do
  read -r client server template <<<"$entry"

  if [[ -n "$filter" ]] && ! echo " $filter " | grep -q " $client "; then
    continue
  fi

  if [[ ! -f "$template" ]]; then
    echo "[$(date +%H:%M:%S)] SKIP $client: template $template missing"
    RESULTS["${server}_${client}"]="SKIP (no template)"
    continue
  fi

  name="${server,,}_${client,,}_minidomainnet"
  config_path="${CONFIG_DIR}/${name}.json"
  log_path="${LOG_DIR}/${name}.log"

  # Patch the template: swap dataset + keep PACS convention (10 clients, IID).
  python3 - <<PY
import json, os
with open("$template") as f:
    c = json.load(f)
c["id"] = "$name"
c["dataset"] = "DomainNet"
c["split_scheme"] = "minidomainnet"
c["data_path"] = "resources/domainnet_v1.0/"
c["dataset_path"] = "resources/"
override = os.environ.get("ROUNDS")
if override:
    c["num_rounds"] = int(override)
group = os.environ.get("WANDB_GROUP", "").strip()
if group:
    c["wandb_group"] = group
project = os.environ.get("WANDB_PROJECT_NAME", "").strip()
if project:
    c["wandb_project"] = project
# leave num_clients / iid / n_groups_per_batch / num_rounds / lr / etc. at template values
with open("$config_path", "w") as f:
    json.dump(c, f, indent=2)
PY

  echo "[$(date +%H:%M:%S)] === $name === (template=$template)"
  start=$(date +%s)
  if python3 main.py --config_file "$config_path" $WANDB_FLAG \
      > "$log_path" 2>&1; then
    end=$(date +%s)
    elapsed=$((end-start))
    # extract final-round best metric, if present
    best=$(grep -oE "best_id_val_acc: [0-9.]+" "$log_path" | tail -1 | awk '{print $2}')
    lodo=$(grep -oE "best_lodo_val_acc: [0-9.]+" "$log_path" | tail -1 | awk '{print $2}')
    RESULTS["${server}_${client}"]="OK  ${elapsed}s  best_id_val=${best:-n/a}  best_lodo_val=${lodo:-n/a}"
    echo "[$(date +%H:%M:%S)] === $name OK in ${elapsed}s (best_id_val=${best:-n/a}, best_lodo_val=${lodo:-n/a}) ==="
  else
    end=$(date +%s)
    elapsed=$((end-start))
    RESULTS["${server}_${client}"]="FAIL after ${elapsed}s (see $log_path)"
    echo "[$(date +%H:%M:%S)] === $name FAILED after ${elapsed}s (see $log_path) ==="
  fi
done

echo ""
echo "================ SUMMARY ================"
for k in "${!RESULTS[@]}"; do
  printf "%-22s %s\n" "$k" "${RESULTS[$k]}"
done | sort
echo "========================================="
echo "logs:    $LOG_DIR/*.log"
echo "configs: $CONFIG_DIR/"
