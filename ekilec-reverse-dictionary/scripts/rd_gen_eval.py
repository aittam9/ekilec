import json
import os
import re 
import sys
import time
import argparse
from unicodedata import normalize
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from rapidfuzz.distance.JaroWinkler import normalized_similarity as jw_sim 
from rapidfuzz.distance.Levenshtein import normalized_similarity as lev_sim


def normalize(text):
        text = re.sub(r'\s+', ' ', text)  
        return re.sub(r'[^\w\s]', '', text.lower()) 

def normalized_exact_match(generated_definition, reference_definition):
    # Normalize the definitions by converting to lowercase and removing punctuation
    return normalize(generated_definition) == normalize(reference_definition)

OUTPUT_DIR = "../results/rd_gen_eval"
INPUT_DIR = "../results/gen_rd"
CACHE_DIR = "/extra/mattia.proietti/hf_models"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the generated definitions")
    parser.add_argument("--input_file", type=str, required=True, choices = os.listdir(INPUT_DIR), help="Path to the input file containing the generated definitions")
    parser.add_argument("--output_file", type=str, required=False, help="Path to the output file where the evaluation results will be saved")
    args = parser.parse_args()

    # Load the input data
    input_path = os.path.join(INPUT_DIR, args.input_file)
    
    data = pd.read_csv(input_path).to_dict(orient='records')

    # Initialize the sentence transformer model
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', cache_folder = CACHE_DIR, device = 'cuda:0')

    # Prepare lists to store evaluation results
    nem_scores = []
    jaro_winkler_scores = []
    levenshtein_scores = []
    semantic_similarity_scores = []
    

    for item in tqdm(data):
        gold_term = normalize(item['gold'])
        predicted_term = normalize(item['predicted'])

        # Compute normalized exact match
        nem_score = normalized_exact_match(predicted_term, gold_term)
        nem_scores.append(nem_score)

        # Compute Jaro-Winkler similarity
        jw_score = jw_sim(predicted_term, gold_term)
        jaro_winkler_scores.append(round(jw_score, 3))

        # compute Levenshtein similarity
        lev_score = lev_sim(predicted_term, gold_term)
        levenshtein_scores.append(round(lev_score, 3))
        # Compute cosine similarity using sentence embeddings
        predicted_embedding = model.encode(predicted_term)
        gold_embedding = model.encode(gold_term)
        semantic_similarity = model.similarity(predicted_embedding, gold_embedding)
        semantic_similarity_scores.append(semantic_similarity.detach().cpu().numpy().round(3).item())

    # Save the evaluation results to a CSV file
    df = pd.DataFrame({
        'gold': [item['gold'] for item in data],
        'predicted': [item['predicted'] for item in data],
        'nem_score': nem_scores,
        'jaro_winkler_score': jaro_winkler_scores,
        'levenshtein_score': levenshtein_scores,
        'SBERT': semantic_similarity_scores
    })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, args.input_file.replace(".tsv", "_rd_eval.tsv"))
    df.to_csv(output_file, index=False)



    # save summary of the evaluation results to a separate CSV file
    row = {"model": args.input_file.replace("_rd.csv", ""), 
           "nem_score": np.mean(nem_scores).round(2).item(),  
           "jaro_winkler_score": np.mean(jaro_winkler_scores).round(2).item(), 
           "levenshtein_score": np.mean(levenshtein_scores).round(2).item(),
           "SBERT": np.mean(semantic_similarity_scores).round(2).item()}
    
    if not os.path.exists(os.path.join(OUTPUT_DIR, "rd_gen_eval_summary.tsv")):
        summary_df = pd.DataFrame(columns=["model", "nem_score", "jaro_winkler_score", "levenshtein_score", "SBERT"])
    else:
        summary_df = pd.read_csv(os.path.join(OUTPUT_DIR, "rd_gen_eval_summary.tsv"), sep='\t')   
    summary_df = pd.concat([summary_df, pd.DataFrame([row])], ignore_index=True)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "rd_gen_eval_summary.tsv"), sep='\t', index=False)