# #!/usr/bin/env bash

# set -euo pipefail

# DATASET="${1:-rd_dataset_formatted.json}"
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# mapfile -t models < <(
#   cd "$SCRIPT_DIR" && python - <<'PY'
# import sys
# sys.path.append("../../common_utils")
# from models import OPEN_ITA_MODELS
# for model_name in OPEN_ITA_MODELS.keys():
#     print(model_name)
# PY
# )

# for model in "${models[@]}"; do
#   echo "Running model: $model on dataset: $DATASET"
#   cd "$SCRIPT_DIR" && python get_answers_rd_local.py --model "$model" --data "$DATASET"
# done

#!/bin/bash
cd "$(dirname "$0")"

MODE=${1:-local}

if [ "$MODE" = "api" ]; then
    MODELS=$(python3 -c "
import sys
sys.path.append('../../common_utils')
from models import API_MODELS
for model_id in API_MODELS.keys():
    print(model_id)
")
    echo "Running with API models..."
      for model in $MODELS; do
        echo "Running with model: $model"
        python get_answers_rd_api.py --model "$model"
      done


elif [ "$MODE" = "local" ]; then
    MODELS=$(python3 -c "
import sys
sys.path.append('../../common_utils')
from models import OPEN_ITA_MODELS
for model_id in OPEN_ITA_MODELS.keys():
    print(model_id)
")
    echo "Running with local models..."

    for model in $MODELS; do
        echo "Running with model: $model"
        python get_answers_rd_local.py --model "$model"
    done
    echo "All local model runs completed."
else
    echo "Invalid mode: $MODE. Use 'local' or 'api'"
    exit 1
fi

