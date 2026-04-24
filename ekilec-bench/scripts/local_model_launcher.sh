# !/bin/bash
#check correct script usage
if [ $# -ne 3 ]; then
    echo "Usage: $0 <model_name> <shuffle> <data>"
    exit 1
fi

model=$1
start=$2
data=$3
# Validate that argument is a positive integer
if ! [[ "$start" =~ ^[0-4]+$ ]]; then
    echo "Error: argument must be a positive integer between 1 and 5."
    exit 1
fi

# For loop from start to 5
for i in $(seq $start 4); do
  echo "Launched shuffle $i with model $model on data $data"
  python3 get_local_answers.py --model $model --shuffle $i --data $data
done