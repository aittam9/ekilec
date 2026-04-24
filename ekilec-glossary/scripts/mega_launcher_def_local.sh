models=("mistral-7b-instruct") #("minerva-7b-inst" "llama-3-8b-instruct" "llamantino-3-anita" "cerbero-7b-open-chat" "mistral-7b-instruct" ) # add here models to run iteratively on all datasets, e.g. "llama-3-8b-instruct" "mistral-7b-instruct" 
batches=("1" "2" "3" "4" "5")

for m in "${models[@]}"; do
  for b in "${batches[@]}"; do
    python generate_definitions_local.py --model $m  --batch $b 
  done
done

