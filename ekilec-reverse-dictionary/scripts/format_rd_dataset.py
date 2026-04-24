import json
import os
import sys

sys.path.append("../common_utils")

from utils import shuffle_data

DATA_DIR = "./data/rd_dataset"
OUTPUT_DIR = "./data/rd_dataset"

def main() -> None:
    
    input_path = os.path.join(DATA_DIR, "rd_dataset_v2.json")
    output_path = os.path.join(OUTPUT_DIR, "rd_dataset_formatted.json")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    data_for_shuffle = {
        f"Question {i}": {
            "question_text": item["definition"],
            "choices": list(item["possible_terms"]),
            "correct_answer": item["correct_term"],
        }
        for i, item in enumerate(raw_data)
    }

    formatted = shuffle_data(data_for_shuffle)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
