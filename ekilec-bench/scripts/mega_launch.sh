
models=("llama-3-8b-instruct") # add here models to run iteratively on all datasets, e.g. "llama-3-8b-instruct" "mistral-7b-instruct" 
datasets=("diritto_privato_ga" "diritto_amm_ga" "proc_civile_ga" "dir_proc_amm_ga")

for m in "${models[@]}"; do
  for d in "${datasets[@]}"; do
    bash local_model_launcher.sh $m 1 $d 
  done
done

