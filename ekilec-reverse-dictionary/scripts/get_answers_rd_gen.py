
import os
import sys
import json
import argparse
import time
import pandas as pd
from tqdm import tqdm

import outlines
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import torch
logging.set_verbosity_error()

import sys
sys.path.append("../../common_utils")
from models import OPEN_ITA_MODELS, API_MODELS
from utils import chat

# from pydotenv import load_dotenv
# load_dotenv("../common_utils/.env")

CACHE_DIR = "/extra/mattia.proietti/hf_models"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


PROMPT = """ Sei un esperto di Giurisprudenza Italiana. Fornisci il concetto più adatto  che corrisponde alla seguente definizione.
**Il concetto può essere costituito da una o più parole**, ma non deve contenere spiegazioni o commenti.
**Devi restituire solo il concetto**, senza spiegazioni o commenti. 
Definizione: {definition}\n\nRisposta:

"""
# Se non conosci la risposta, restituisci una stringa vuota.
def build_prompt(definition: str) -> str:
	# return (


	# 	"Provide the single best term (only the term, no explanation) that matches the following definition.\n\n"
	# 	f"Definition: {definition}\n\nAnswer:" 
	# )
	return PROMPT.replace("{definition}", definition)


def clean_prediction(pred: str) -> str:
	if not pred:
		return ""
	# take first line, strip, remove surrounding quotes and trailing punctuation
	line = pred.splitlines()[0].strip()
	line = line.strip('"')
	# remove trailing periods if they appear
	if line.endswith('.'):
		line = line[:-1].strip()
	return line


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", "-d", default="../rd_dataset/rd_dataset.json", help="Path to rd_dataset.json")
	parser.add_argument("--model", "-m", required=True, help="Model name: local (from OPEN_ITA_MODELS) or API (from API_MODELS)")
	parser.add_argument("--cache-dir", default=CACHE_DIR, help="HF cache dir (optional)")
	parser.add_argument("--device", default=DEVICE, help="Device (e.g. cuda:0). If omitted uses cuda if available")
	parser.add_argument("--outdir", default="../results/gen_rd", help="Output directory for CSV results")
	parser.add_argument("--max-tokens", type=int, default=16, help="Max new tokens for generation")
	args = parser.parse_args()

	data_path = args.data
	model_name = args.model
	cache_dir = args.cache_dir
	device = args.device

	# Determine if model is local or API
	is_local = model_name in OPEN_ITA_MODELS
	is_api = model_name in API_MODELS

	if not is_local and not is_api:
		print(f"Error: Model '{model_name}' not found in OPEN_ITA_MODELS or API_MODELS")
		return

	model_id = OPEN_ITA_MODELS[model_name] if is_local else API_MODELS[model_name]
	model_type = "local" if is_local else "API"

	os.makedirs(args.outdir, exist_ok=True)

	with open(data_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	print(f"Loaded {len(data)} items from {data_path}")
	print(f"Using {model_type} model: {model_id}")

	# Load model only if local
	if is_local:
		print(f"Loading model (device={device})")
		tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
		model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir).to(device)
		outline_model = outlines.from_transformers(model, tokenizer)

	results = []
	start = time.time()
	for item in tqdm(data):
		definition = item.get("definition", "")
		gold = item.get("correct_term", "")
		prompt = build_prompt(definition)
		
		if is_local:
			# Generate using local model
			raw = outline_model.generate(prompt, max_new_tokens=args.max_tokens, temperature=0., do_sample=False)
			pred = clean_prediction(raw)
		else:
			max_tokens = 1000 if "gpt-5" in  model_id or "gemini" in model_id else args.max_tokens
			raw = chat(prompt, model_id, max_tokens=max_tokens)
			# pred = clean_prediction(raw) if raw else ""
			pred = raw

		results.append({"definition": definition, "gold": gold, "predicted": pred})

	df = pd.DataFrame(results)
	model_short_name = model_id.split('/')[-1]
	outpath = os.path.join(args.outdir, f"{model_short_name}_rd.csv")
	df.to_csv(outpath, index=False)

	elapsed = time.time() - start
	print(f"Saved {len(df)} rows to {outpath} — took {elapsed:.2f}s")


if __name__ == "__main__":
	main()
