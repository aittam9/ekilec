import re
import unicodedata
import json
import os
import argparse
import random
from typing import Literal, List, Tuple, Dict


from sentence_transformers import SentenceTransformer

def normalize_text(text: str) -> str:
    # Lowercase, strip accents, remove punctuation, and collapse spaces.
    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def infer_term_morphology(term: str) -> tuple[str, str]:
    """Infer (gender, number) with lightweight Italian heuristics.

    Returns values among:
    - gender: "m", "f", "u" (unknown)
    - number: "sg", "pl", "u" (unknown)
    """
    text = normalize_text(term)
    tokens = [t for t in text.split() if t]
    if not tokens:
        return ("u", "u")

    first = tokens[0]
    article_hint = {
        "il": ("m", "sg"),
        "lo": ("m", "sg"),
        "l": ("u", "sg"),
        "la": ("f", "sg"),
        "i": ("m", "pl"),
        "gli": ("m", "pl"),
        "le": ("f", "pl"),
        "un": ("m", "sg"),
        "uno": ("m", "sg"),
        "una": ("f", "sg"),
    }
    if first in article_hint:
        return article_hint[first]

    # Pick first lexical token (skip common function words).
    skip_tokens = {
        "di", "del", "dello", "della", "dei", "degli", "delle",
        "a", "al", "allo", "alla", "ai", "agli", "alle",
        "da", "dal", "dallo", "dalla", "dai", "dagli", "dalle",
        "in", "nel", "nello", "nella", "nei", "negli", "nelle",
        "con", "su", "sul", "sullo", "sulla", "sui", "sugli", "sulle",
        "per", "tra", "fra", "e", "o",
    }
    lex = next((t for t in tokens if t not in skip_tokens), tokens[0])

    ending = lex[-1]
    if ending == "o":
        return ("m", "sg")
    if ending == "a":
        return ("f", "sg")
    if ending == "i":
        return ("m", "pl")
    if ending == "e":
        return ("f", "pl")
    return ("u", "u")


def same_morphology(term_a: str, term_b: str) -> bool:
    ga, na = infer_term_morphology(term_a)
    gb, nb = infer_term_morphology(term_b)
    if ga == "u" or gb == "u" or na == "u" or nb == "u":
        return False
    return ga == gb and na == nb


def pad_distractors(
    term: str,
    current_distractors: List[str],
    all_terms: List[str],
    rng: random.Random,
    target_count: int,
    match_gender_number: bool,
) -> List[str]:
    missing_count = target_count - len(current_distractors)
    if missing_count <= 0:
        return []

    excluded = set(current_distractors)
    excluded.add(term)
    candidates = [cand for cand in all_terms if cand not in excluded]

    if match_gender_number:
        morph_candidates = [cand for cand in candidates if same_morphology(term, cand)]
        if morph_candidates:
            candidates = morph_candidates

    if not candidates:
        return []

    if len(candidates) >= missing_count:
        return rng.sample(candidates, missing_count)

    padded = rng.sample(candidates, len(candidates))
    while len(padded) < missing_count:
        padded.append(rng.choice(candidates))
    return padded



def term_in_definition(term: str, definition: str, mode: Literal["strict", "loose"] = "loose") -> bool:
    term_n = normalize_text(term)
    def_n = normalize_text(definition)
    if not term_n or not def_n:
        return False

    # Direct containment (exact phrase or partial substring).
    if mode == "loose":
        if term_n in def_n:
            return True


    elif mode == "strict":
        if term_n == def_n:
            return True
        # Token-level partial overlap (ignore very short tokens to reduce noise).
        term_tokens = [t for t in term_n.split() if len(t) >= 4 ]
        def_tokens = [t for t in def_n.split() if len(t) >= 4]
        for t in term_tokens:
            for d in def_tokens:
                if t in d or d in t:
                    return True

    return False


def compute_similarity_matrix(model: SentenceTransformer, sentences: List[str]):
    embeddings = model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
    return model.similarity(embeddings, embeddings)


BATCHES_PATH = "../../ekilec-glossary/data/batches4search"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build retrieval-driven MCQ-style glossary dataset from local JSON batches."
    )
    parser.add_argument(
        "--mode",
        choices=["strict", "loose"],
        default="loose",
        help="Filtering mode for removing entries where the true term appears in its definition.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k most similar entries used as distractor source pool.",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=3,
        help="Initial number of distractor terms sampled from the top-k pool before padding to three total.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data/rd_dataset_v2.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--similarity-on",
        choices=["definitions", "terms"],
        default="definitions",
        help="Use similarity between definitions (default) or between terms to select candidate distractors.",
    )
    parser.add_argument(
        "--match-gender-number",
        action="store_true",
        help="Only sample distractors with same inferred gender/number as the correct term.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # load data
    all_data = []
    for file in os.listdir(BATCHES_PATH):
        if file.endswith(".json"):
            with open(os.path.join(BATCHES_PATH, file), "r") as f:
                all_data.extend(json.load(f))

    # Keep only valid term/definition pairs before computing similarities.
    all_data = [
        e
        for e in all_data
        if str(e.get("term", "")).strip() and str(e.get("term_definition", "")).strip()
    ]

    terms = [str(e["term"]).strip() for e in all_data]
    definitions = [str(e["term_definition"]).strip() for e in all_data]
    unique_terms = list(dict.fromkeys(terms))

    if len(definitions) < 2:
        raise ValueError("Need at least 2 valid entries to build distractors.")

    # compute similarity scores
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    similarity_texts = definitions if args.similarity_on == "definitions" else terms
    sim_matrix = compute_similarity_matrix(model, similarity_texts)

    dataset = []
    n = len(definitions)
    top_k = min(args.top_k, n - 1)
    target_distractors = 3
    padded_entries = 0

    for i, (term, definition) in enumerate(zip(terms, definitions)):
        row = sim_matrix[i].clone()
        row[i] = float("-inf")  # exclude self

        top_indices = row.topk(k=top_k).indices.tolist()

        # Build a unique distractor pool from terms of the top-k similar definitions.
        distractor_pool = []
        seen = {term}
        for idx in top_indices:
            cand = terms[idx]
            if cand not in seen:
                distractor_pool.append(cand)
                seen.add(cand)

        if args.match_gender_number:
            distractor_pool = [cand for cand in distractor_pool if same_morphology(term, cand)]

        sample_size = min(args.sample_n, target_distractors, len(distractor_pool))
        sampled_terms = rng.sample(distractor_pool, sample_size) if sample_size > 0 else []

        if len(sampled_terms) < target_distractors:
            padded_entries += 1
            sampled_terms.extend(
                pad_distractors(
                    term=term,
                    current_distractors=sampled_terms,
                    all_terms=unique_terms,
                    rng=rng,
                    target_count=target_distractors,
                    match_gender_number=args.match_gender_number,
                )
            )

        sampled_terms = sampled_terms[:target_distractors]

        options = sampled_terms + [term]
        rng.shuffle(options)

        dataset.append(
            {
                "definition": definition,
                "possible_terms": options,
                "correct_term": term,
            }
        )

    # final filter: remove entries where the correct term appears in the definition text
    mode = args.mode
    filtered_dataset = [
        e
        for e in dataset
        if not term_in_definition(e["correct_term"], e["definition"], mode=mode)
    ]

    removed_count = len(dataset) - len(filtered_dataset)
    print(f"Distractor retrieval similarity: {args.similarity_on}")
    print(f"Morphology constraint: {args.match_gender_number}")
    print(f"Filtering strategy: {mode}")
    print(f"Entries padded to 4 choices: {padded_entries}")
    print(f"Removed: {removed_count} / {len(dataset)}")
    print(f"Remaining entries: {len(filtered_dataset)}")

    output_path = args.output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(filtered_dataset, f, ensure_ascii=False, indent=2)

    print(f"Saved dataset to: {output_path}")


