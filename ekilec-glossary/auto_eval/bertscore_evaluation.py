import argparse
import os 
from typing import Dict, List, Literal, Optional
from transformers.utils import logging
logging.set_verbosity_error() # to hide transformers warnings


import pandas as pd 
import numpy as np
from bert_score import BERTScorer

# # Monkey-patch to fix missing method in XLMRobertaTokenizer
# from transformers import XLMRobertaTokenizer

# def _build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
#     """
#     Build model inputs from a sequence or a pair of sequence for sequence classification tasks
#     by concatenating and adding special tokens.
#     """
#     cls = [self.cls_token_id]
#     sep = [self.sep_token_id]
#     if token_ids_1 is None:
#         return cls + token_ids_0 + sep
#     return cls + token_ids_0 + sep + token_ids_1 + sep

# # Apply monkey-patch if method doesn't exist
# if not hasattr(XLMRobertaTokenizer, 'build_inputs_with_special_tokens'):
#     XLMRobertaTokenizer.build_inputs_with_special_tokens = _build_inputs_with_special_tokens




class CustomScorer(BERTScorer):

    """
    A custom BERTScorer that computes baseline scores on random pairs and rescales the scores accordingly.
    :params
    candidates: list of candidate sentences
    references: list of reference sentences
    lang: language code (default "it" for Italian)
    model_type: the BERT model to use (default "dlicari/Italian-Legal-BERT")
    num_layers: number of layers to use from the BERT model (default 12)
    cache_dir: directory to cache the BERT model (default None)
    rescale_with_baseline: whether to rescale scores with baseline (default False)
    return
    """
    def __init__(self,
                # candidates: list,
                # references: list,
                lang: str = "it",
                model_type: str = "dlicari/Italian-Legal-BERT",  #"dlicari/Italian-Legal-BERT" "Qwen/Qwen3-Embedding-0.6B" "BAAI/bge-large-en-v1.5"
                num_layers: int = 12,
                cache_dir: str = None,
                rescale_with_baseline = False,
                **kwargs):
        
        super().__init__(lang=lang,
                        model_type=model_type,
                        num_layers=num_layers,
                        rescale_with_baseline = rescale_with_baseline,
                        
                        **kwargs)

        
        # self.candidates = candidates
        # self.references = references
        self.apply_rescale_with_baseline = rescale_with_baseline
        self.baseline_scores = {}


    def score_baseline(self)-> Dict[str, np.ndarray]:
        print(f"Computing baselines on random pairs")
        shuffled_references = self.references.copy()
        np.random.shuffle(shuffled_references)  #maybe shuffle only one between ref and candidates
        #np.random.shuffle(candidate)
        P, R, F1 = super().score(self.candidates, shuffled_references)
        self.baseline_scores = {"baseline_P": P, "baseline_R": R, "baseline_F1": F1}
        return self.baseline_scores


    def rescale(self, target_value, baseline)-> np.ndarray:
        return (target_value - baseline) / (1 - baseline)
    

    def score_rescaled(self, candidates, references) -> Dict[str,np.ndarray]:
        # get unscaled scores
        self.candidates = candidates
        self.references = references
        P, R, F1 = super().score(self.candidates, self.references)
        
        # get baselines scores
        baseline = self.score_baseline()
        # rescaling values
        P = self.rescale(P, baseline["baseline_P"]).detach().cpu().numpy()
        R = self.rescale(R, baseline["baseline_R"]).detach().cpu().numpy()
        F1 = self.rescale(F1, baseline["baseline_F1"]).detach().cpu().numpy()
        self.rescaled_scores = {"P": P, "R": R, "F1": F1}
        return self.rescaled_scores
    

CACHE_DIR = "/extra/mattia.proietti/hf_models"  
BASE_DATA_DIR = "./results/generation/"
OUTPUT_DIR = "./results/evaluation/bertscore/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-d", default="", help="Path to the tsv file containing the data to evaluate")
    parser.add_argument("--bert_type", "-mt", default= "dlicari/Italian-Legal-BERT", help="The bertscore model to use") #"dlicari/Italian-Legal-BERT" "Qwen/Qwen3-Embedding-0.6B" "BAAI/bge-large-en-v1.5"
    parser.add_argument("--output_dir", "-o", choices=["default", "embedder"], default = "default",  help="Directory to save the evaluation results")
    args = parser.parse_args()


    # instantiate output dir
    if  args.output_dir == "default": 
        model_outdir = os.path.join(OUTPUT_DIR, args.model)
    else:
        model_outdir = os.path.join(OUTPUT_DIR, args.bert_type.split("/")[-1], args.model)
    print(f"Saving results to {model_outdir}")
    
    os.makedirs(model_outdir, exist_ok=True)
            

    # open and concatenate all data 
    print(f"Loading data from {os.path.join(BASE_DATA_DIR, args.model)}")
    model_indir = os.path.join(BASE_DATA_DIR, args.model)
    all_data = []
    for file in os.listdir(model_indir):
        if file.endswith(".tsv"):
            file_path = os.path.join(model_indir, file)
            df = pd.read_csv(file_path, sep = "\t")
            all_data.append(df)
    all_data = pd.concat(all_data, ignore_index=True)

    candidates = all_data['model_response'].tolist()
    references = all_data['definition'].tolist()

    #instantiate custom scorer
    scorer = CustomScorer(#candidates=candidates,
                          #references=references,
                          model_type=args.bert_type,
                          lang="it",
                          cache_dir=CACHE_DIR,
                          rescale_with_baseline=False)
    
    print(scorer.model_type)
    
    # score results
    print(f"Evaluating...")
    rescaled_scores = scorer.score_rescaled(candidates, references)

    #print(rescaled_scores)
    results_df = pd.DataFrame(rescaled_scores)

    outfile_name = os.path.join(model_outdir, f"bertscore_rescaled_{args.model}.tsv")
    results_df.to_csv(outfile_name, sep = "\t", index = False)
    
    print(f"Saved results to {outfile_name}")
    
    print(f"Average Rescaled BERTScore Values for {args.model}" ) 
    print(f"F1: {np.mean(rescaled_scores['F1']).round(3)}")
    print(f"P: {np.mean(rescaled_scores['P']).round(3)}" )
    print(f"R: {np.mean(rescaled_scores['R']).round(3)}" )
    with open(os.path.join(model_outdir, f"bertscore_summary_{args.model}.txt"), "w") as outfile:
        outfile.write(f"Average Rescaled BERTScore Values for {args.model}\n" ) 
        outfile.write(f"F1: {np.mean(rescaled_scores['F1']).round(3)}\n")
        outfile.write(f"P: {np.mean(rescaled_scores['P']).round(3)}\n" )
        outfile.write(f"R: {np.mean(rescaled_scores['R']).round(3)}\n" )