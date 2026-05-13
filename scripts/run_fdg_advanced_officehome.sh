#!/usr/bin/env bash
#
# Run advanced FedDG baselines on OfficeHome (ResNet-50 backbone, the default)
# under the non-IID, large-client regime that matches run_fdg_advanced_minidomainnet.sh:
#
#     num_clients = 100
#     iid         = 0.0
#     fraction    = 0.2
#     n_groups_per_batch = 1   (each client holds a single domain shard under iid=0)
#     num_workers = 4
#
# Default L2DO baked into resources/office_home_v1.0/metadata.csv:
#     val=art, test=real_world. Train pool = {clipart, product}.
# Train pool has ~7,924 rows -> ~80 rows per client at num_clients=100.
#
# Methods (server -> client):
#   FedAvg    FedAvg          ERM            (vanilla federated baseline)
#   FCDv2     FCDv2Server     FCDv2Client    (gaussian aggregator, resnet50)
#   FedProx   FedAvg          FedProx
#   Scaffold  ScaffoldServer  ScaffoldClient
#   AFL       AFLServer       AFLClient
#   FedADG    FedADGServer    FedADGClient
#   FedSR     FedAvg          FedSR
#   FedGMA    FedGMA          ERM            (FedGMA is a server-side aggregation)
#
# Usage:
#   bash scripts/run_fdg_advanced_officehome.sh                  # all 7, wandb on
#   ROUNDS=30 bash scripts/run_fdg_advanced_officehome.sh        # override num_rounds for all
#   METHODS="FedProx FedSR" bash scripts/run_fdg_advanced_officehome.sh   # subset
#   NO_WANDB=1 bash scripts/run_fdg_advanced_officehome.sh       # disable wandb
#   WANDB_GROUP=my_sweep bash scripts/run_fdg_advanced_officehome.sh

set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

WANDB_FLAG=""
if [[ "${NO_WANDB:-0}" != "0" ]]; then
  WANDB_FLAG="--no_wandb"
fi

WANDB_GROUP="${WANDB_GROUP:-officehome_n100_iid0_frac20}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-FedDG_Benchmark_officehome}"

LOG_DIR="logs/fdg_advanced_officehome"
CONFIG_DIR="${LOG_DIR}/configs"
mkdir -p "$LOG_DIR" "$CONFIG_DIR"

# Identifier:Server:Client:Template
BASELINES=(
  "FedAvg:FedAvg:ERM:configs/erm/fedavg_erm_pacs.json"
  "FCDv2:FCDv2Server:FCDv2Client:configs/fcdv2/fcdv2_pacs_r50_vae.json"
  "FedProx:FedAvg:FedProx:configs/fedprox/fedavg_fedprox_pacs.json"
  "Scaffold:ScaffoldServer:ScaffoldClient:configs/scaffoldclient/scaffoldserver_scaffoldclient_pacs.json"
  "AFL:AFLServer:AFLClient:configs/aflclient/aflserver_aflclient_pacs.json"
  "FedADG:FedADGServer:FedADGClient:configs/fedadgclient/fedadgserver_fedadgclient_pacs.json"
  "FedSR:FedAvg:FedSR:configs/fedsr/fedavg_fedsr_pacs.json"
  "FedGMA:FedGMA:ERM:configs/erm/fedgma_erm_pacs.json"
)

filter="${METHODS:-}"

declare -A RESULTS

for entry in "${BASELINES[@]}"; do
  IFS=":" read -r tag server client template <<<"$entry"

  if [[ -n "$filter" ]] && ! echo " $filter " | grep -q " $tag "; then
    continue
  fi

  if [[ ! -f "$template" ]]; then
    echo "[$(date +%H:%M:%S)] SKIP $tag: template $template missing"
    RESULTS["$tag"]="SKIP (no template)"
    continue
  fi

  name="${tag,,}_officehome_n100_iid0"
  config_path="${CONFIG_DIR}/${name}.json"
  log_path="${LOG_DIR}/${name}.log"

  python3 - "$template" "$config_path" "$name" <<'PY'
import json, os, sys
template, out_path, run_id = sys.argv[1], sys.argv[2], sys.argv[3]
with open(template) as f:
    c = json.load(f)
c["id"] = run_id
c["dataset"] = "OfficeHome"
c["split_scheme"] = "official"
c["data_path"] = "resources/office_home_v1.0/"
c["dataset_path"] = "resources/"
c["num_clients"] = 100
c["iid"] = 0.0
c["fraction"] = 0.2
c["n_groups_per_batch"] = 1
c["num_workers"] = 4
override = os.environ.get("ROUNDS")
if override:
    c["num_rounds"] = int(override)
group = os.environ.get("WANDB_GROUP", "").strip()
if group:
    c["wandb_group"] = group
project = os.environ.get("WANDB_PROJECT_NAME", "").strip()
if project:
    c["wandb_project"] = project
with open(out_path, "w") as f:
    json.dump(c, f, indent=2)
PY

  echo "[$(date +%H:%M:%S)] === $name === (template=$template)"
  start=$(date +%s)
  if python3 main.py --config_file "$config_path" $WANDB_FLAG \
      > "$log_path" 2>&1; then
    end=$(date +%s)
    elapsed=$((end-start))
    best=$(grep -oE "best_id_val_acc: [0-9.]+" "$log_path" | tail -1 | awk '{print $2}')
    lodo=$(grep -oE "best_lodo_val_acc: [0-9.]+" "$log_path" | tail -1 | awk '{print $2}')
    RESULTS["$tag"]="OK  ${elapsed}s  best_id_val=${best:-n/a}  best_lodo_val=${lodo:-n/a}"
    echo "[$(date +%H:%M:%S)] === $tag OK in ${elapsed}s (best_id_val=${best:-n/a}, best_lodo_val=${lodo:-n/a}) ==="
  else
    end=$(date +%s)
    elapsed=$((end-start))
    RESULTS["$tag"]="FAIL after ${elapsed}s (see $log_path)"
    echo "[$(date +%H:%M:%S)] === $tag FAILED after ${elapsed}s (see $log_path) ==="
  fi
done

echo ""
echo "================ SUMMARY ================"
for k in "${!RESULTS[@]}"; do
  printf "%-10s %s\n" "$k" "${RESULTS[$k]}"
done | sort
echo "========================================="
echo "logs:    $LOG_DIR/*.log"
echo "configs: $CONFIG_DIR/"
