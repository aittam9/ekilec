# EKILeC
**E**xplicit **K**nowledge of **I**talian **Le**gal **C**oncept
EKILeC is a collection of experiments for evaluating large language models on Italian legal language tasks. The repository is split into three main workflows:

- `ekilec-bench`: multiple-choice evaluation on legal concepts
- `ekilec-glossary`: glossary/definition generation for legal terms
- `ekilec-reverse-dictionary`: reverse dictionary experiments, where the model predicts a term from its definition

Shared prompts, model registries, and helpers live in `common_utils/`.

## Repository Layout

- `common_utils/models.py`: shared prompt templates and model IDs
- `common_utils/utils.py`: data helpers used across scripts
- `ekilec-bench/scripts/`: evaluation scripts for MCQ-style benchmarks
- `ekilec-glossary/scripts/`: scripts for generating definitions from legal terms
- `ekilec-reverse-dictionary/scripts/`: scripts for building and evaluating the reverse dictionary dataset

## Example Usage

### 1. Benchmark legal MCQs

Run a local open model on one shuffled dataset:

```bash
cd ekilec-bench/scripts
python get_local_answers.py --model llama-3-8b-instruct --data diritto_privato_ga --shuffle 1
```

Run an API model through OpenRouter:

```bash
cd ekilec-bench/scripts
python get_response_openrouter.py --model gpt-4o --data diritto_privato_ga --shuffle 1
```

Batch all benchmark shuffles with a shell launcher:

```bash
cd ekilec-bench/scripts
bash mega_launch.sh
```

### 2. Generate glossary definitions

Generate definitions with a local model:

```bash
cd ekilec-glossary/scripts
python generate_definitions_local.py --model mistral-7b-instruct --batch 1
```

Generate definitions through OpenRouter:

```bash
cd ekilec-glossary/scripts
python generate_definitions_api.py --model gpt-4o --batch 1
```

Run the local launcher across all configured batches:

```bash
cd ekilec-glossary/scripts
bash mega_launcher_def_local.sh
```

### 3. Reverse dictionary experiments

Build the formatted reverse-dictionary dataset:

```bash
cd ekilec-reverse-dictionary/scripts
python format_rd_dataset.py
```

Create the retrieval-based dataset from glossary batches:

```bash
cd ekilec-reverse-dictionary/scripts
python build_rd_dataset.py --output ./data/rd_dataset/rd_dataset.json
```

Evaluate a local model on the reverse-dictionary benchmark:

```bash
cd ekilec-reverse-dictionary/scripts
python get_answers_rd_local.py --model llama-3-8b-instruct --data rd_dataset_formatted.json
```

Run every available local or API model:

```bash
cd ekilec-reverse-dictionary/scripts
bash launch_all_rd_local_models.sh
bash launch_all_rd_api_models.sh
```

## Notes

- OpenRouter-based scripts expect an `OPENROUTER_KEY` environment variable.
- Most scripts write results under their local `results/` directory and print a quick accuracy or completion summary to the terminal.
- Some scripts assume they are launched from their own `scripts/` directory, as shown above.