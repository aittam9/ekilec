import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from rouge_score import rouge_scorer
import sacrebleu


CACHE_DIR = "/extra/mattia.proietti/hf_models"
BASE_DATA_DIR = "./results/generation/"
OUTPUT_DIR = "./results/evaluation/rouge_bleu/"


class RougeBleuEvaluator:
    """
    Class to compute ROUGE and BLEU scores for candidate and reference texts.
    """ 

    def __init__(self, candidates: List[str], references: List[str]):
        self.rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"])
        self.bleu_scorer = sacrebleu.BLEU(effective_order=True)
        self.candidates = candidates
        self.references = references

    def compute_rouge(self) -> Dict[str, List[float]]:
        
        rouge1, rouge2, rougeL, rougeLsum = [], [], [], []
        for cand, ref in zip(self.candidates, self.references):
                scores = self.rouge_scorer.score(ref, cand)
                rouge1.append(scores["rouge1"].fmeasure)
                rouge2.append(scores["rouge2"].fmeasure)
                rougeL.append(scores["rougeL"].fmeasure)
                rougeLsum.append(scores["rougeLsum"].fmeasure)
        return {
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
            "rougeLsum": rougeLsum,
        }


    def compute_bleu(self) -> Dict[str, List[float]]:
        bleu_scores = []
        for cand, ref in zip(self.candidates, self.references):
            score = self.bleu_scorer.sentence_score(cand, [ref]).score
            bleu_scores.append(score)
        return {"bleu": bleu_scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-d", default="", help="Path to the tsv folder containing the data to evaluate")
    args = parser.parse_args()

    model_outdir = os.path.join(OUTPUT_DIR, args.model)
    if not os.path.exists(model_outdir):
        os.makedirs(model_outdir)

    print(f"Loading data from {os.path.join(BASE_DATA_DIR, args.model)}")
    model_indir = os.path.join(BASE_DATA_DIR, args.model)
    all_data = []
    for file in os.listdir(model_indir):
        if file.endswith(".tsv"):
            file_path = os.path.join(model_indir, file)
            df = pd.read_csv(file_path, sep="\t")
            all_data.append(df)
    all_data = pd.concat(all_data, ignore_index=True)

    candidates = all_data["model_response"].tolist()
    references = all_data["definition"].tolist()

    evaluator = RougeBleuEvaluator(candidates, references)
    print("Evaluating ROUGE...")
    rouge_scores = evaluator.compute_rouge()    

    print("Evaluating BLEU...")
    bleu_scores = evaluator.compute_bleu()

    results = {**rouge_scores, **bleu_scores}
    results_df = pd.DataFrame(results)

    outfile_name = os.path.join(model_outdir, f"rouge_bleu_{args.model}.tsv")
    results_df.to_csv(outfile_name, sep="\t", index=False)

    print(f"Saved results to {outfile_name}")

    print(f"Average Metrics for {args.model}")
    print(f"ROUGE-1: {np.mean(rouge_scores['rouge1']).round(3)}")
    print(f"ROUGE-2: {np.mean(rouge_scores['rouge2']).round(3)}")
    print(f"ROUGE-L: {np.mean(rouge_scores['rougeL']).round(3)}")
    print(f"ROUGE-Lsum: {np.mean(rouge_scores['rougeLsum']).round(3)}")
    print(f"BLEU: {np.mean(bleu_scores['bleu']).round(3)}")

    with open(os.path.join(model_outdir, f"rouge_bleu_summary_{args.model}.txt"), "w") as outfile:
        outfile.write(f"Average Metrics for {args.model}\n")
        outfile.write(f"ROUGE-1: {np.mean(rouge_scores['rouge1']).round(3)}\n")
        outfile.write(f"ROUGE-2: {np.mean(rouge_scores['rouge2']).round(3)}\n")
        outfile.write(f"ROUGE-L: {np.mean(rouge_scores['rougeL']).round(3)}\n")
        outfile.write(f"ROUGE-Lsum: {np.mean(rouge_scores['rougeLsum']).round(3)}\n")
        outfile.write(f"BLEU: {np.mean(bleu_scores['bleu']).round(3)}\n")
