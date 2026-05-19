#!/bin/bash

INPUT_DIR="../results/gen_rd"

for file in "$INPUT_DIR"/*.csv; do
    filename=$(basename "$file")
   
    
    echo "Processing: $filename"
    python rd_gen_eval.py --input_file "$filename" 
done

echo "All files processed!"
