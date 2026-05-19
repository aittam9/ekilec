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
elif [ "$MODE" = "local" ]; then
    MODELS=$(python3 -c "
import sys
sys.path.append('../../common_utils')
from models import OPEN_ITA_MODELS
for model_id in OPEN_ITA_MODELS.keys():
    print(model_id)
")
    echo "Running with local models..."
else
    echo "Invalid mode: $MODE. Use 'local' or 'api'"
    exit 1
fi

for model in $MODELS; do
    echo "Running with model: $model"
    python get_answers_rd_gen.py --model "$model"
done
