import argparse
import os
import pickle
from typing import Dict, List
import json
import time 

import numpy as np
import pandas as pd

from bertscore_evaluation import CustomScorer
from rouge_bleu_evaluation import RougeBleuEvaluator


CACHE_DIR = "/extra/mattia.proietti/hf_models"
BASE_DATA_DIR = "./results/generation/"
OUTPUT_DIR = "./results/evaluation/test/cross_model/"


def load_model_data(model_name: str) -> pd.DataFrame:
    model_indir = os.path.join(BASE_DATA_DIR, model_name)
    if not os.path.isdir(model_indir):
        raise FileNotFoundError(f"Missing generation directory for model '{model_name}': {model_indir}")

    all_data = []
    for file_name in sorted(os.listdir(model_indir)):
        if file_name.endswith(".tsv"):
            file_path = os.path.join(model_indir, file_name)
            df = pd.read_csv(file_path, sep="\t")
            all_data.append(df)

    if not all_data:
        raise ValueError(f"No TSV files found in {model_indir}")

    return pd.concat(all_data, ignore_index=True)


def compute_pairwise_metric_matrices(
    big_model_matrix_df: pd.DataFrame,
    bert_scorer: CustomScorer,
) -> Dict[str, pd.DataFrame]:
    model_names = big_model_matrix_df.columns.tolist()
    metric_names = ["bertscore_f1", "rouge1", "rouge2", "rougeL", "rougeLsum", "bleu"]
    metric_matrices = {
        metric: pd.DataFrame(np.nan, index=model_names, columns=model_names)
        for metric in metric_names
    }

    for model_i in model_names:
        candidates = big_model_matrix_df[model_i].fillna("").astype(str).tolist()
        for model_j in model_names:
            references = big_model_matrix_df[model_j].fillna("").astype(str).tolist()

            rouge_bleu_evaluator = RougeBleuEvaluator(candidates=candidates, references=references)
            rouge_scores = rouge_bleu_evaluator.compute_rouge()
            bleu_scores = rouge_bleu_evaluator.compute_bleu()

            metric_matrices["rouge1"].loc[model_i, model_j] = float(np.mean(rouge_scores["rouge1"]))
            metric_matrices["rouge2"].loc[model_i, model_j] = float(np.mean(rouge_scores["rouge2"]))
            metric_matrices["rougeL"].loc[model_i, model_j] = float(np.mean(rouge_scores["rougeL"]))
            metric_matrices["rougeLsum"].loc[model_i, model_j] = float(np.mean(rouge_scores["rougeLsum"]))
            metric_matrices["bleu"].loc[model_i, model_j] = float(np.mean(bleu_scores["bleu"]))

         
            bert_scores = bert_scorer.score_rescaled(candidates, references)
            metric_matrices["bertscore_f1"].loc[model_i, model_j] = float(np.mean(bert_scores["F1"]))

            print(f"Computed metrics for pair ({model_i}, {model_j})")

    return metric_matrices



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bert_type",
        "-mt",
        default= "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", #dlicari/Italian-Legal-BERT",
        help="BERT model used by BERTScore",
    )
    
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start = time.monotonic()
    # instantiate models
    models = sorted(os.listdir(BASE_DATA_DIR))
    if not models:
        raise ValueError("No models found after filtering.")

    #instantiate bif matrix with terms and model responses
    big_model_matrix: Dict[str, List[str]] = {}
    first_model_data = None

    for model_name in models:
        print(f"Loading data from {os.path.join(BASE_DATA_DIR, model_name)}")
        model_data = load_model_data(model_name)
        print(model_data.shape)

        if first_model_data is None:
            first_model_data = model_data
            big_model_matrix["terms"] = first_model_data["term"].astype(str).tolist()

        big_model_matrix[model_name] = model_data["model_response"].astype(str).tolist()
    

    big_model_matrix_df = pd.DataFrame(big_model_matrix).set_index("terms")
    print(f"Models in matrix: {big_model_matrix_df.columns.tolist()}")

    print(f"\nLoading BERTScore model '{args.bert_type}'...")
    bert_scorer = CustomScorer(
                # candidates=candidates,
                # references=references,
                lang="it",
                model_type=args.bert_type,
                cache_dir=CACHE_DIR,
                rescale_with_baseline=False,
            )
    
    #store results matices
    metric_matrices = compute_pairwise_metric_matrices(
        big_model_matrix_df=big_model_matrix_df,
        bert_scorer=bert_scorer,
    )

    out_dict_path = os.path.join(OUTPUT_DIR, "pairwise_metric_matrices.pkl")
    with open(out_dict_path, "wb") as out_file:
        pickle.dump(metric_matrices, out_file)

    for metric_name, metric_df in metric_matrices.items():
        metric_outfile = os.path.join(OUTPUT_DIR, f"pairwise_{metric_name}_matrix.tsv")
        metric_df.to_csv(metric_outfile, sep="\t")

    print(f"Saved metric dictionary to {out_dict_path}")
    end = time.monotonic()
    elapsed = divmod(end - start, 60)
    print(f"Total time taken: {elapsed[0]} minutes and {elapsed[1]:.2f} seconds")
