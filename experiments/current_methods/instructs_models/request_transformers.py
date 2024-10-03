import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

def request_model(
        query:str, 
        max_year:int,
        model:AutoModelForCausalLM, 
        tokenizer:AutoTokenizer,
    ) -> str:
    query = query.replace("Give me a reading list", f"Give me a reading list of 20 articles up to {max_year}")
    messages = [{"role": "user", "content": query}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(accelerator.device)
    out = model.generate(inputs, max_length=4096)
    return tokenizer.decode(out[0])


def process_model_requests(
        model:AutoModelForCausalLM, 
        tokenizer:AutoTokenizer, 
        annotations_path:str, 
        output_path:str
    )->None:
    
    for annotator_num in [1, 2, 3]:
        annotator_queries = pd.read_csv(f"{annotations_path}annotation_{annotator_num}.csv")[["id", "year", "query_sentence"]].replace(np.nan, None).to_dict(orient='records')
        for query in tqdm(annotator_queries, desc=f"A{annotator_num}"):

            file_path = Path(f"{output_path}/annotator{annotator_num}/{query['id']}.md")
            if not file_path.is_file():
                preds_annot = request_model(query["query_sentence"], query["year"], model, tokenizer)
                os.makedirs(file_path, exist_ok=True)
                with open(file_path, "w") as file:
                    file.write(preds_annot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Request HuggingFace model for the generation of a reading list')
    parser.add_argument('--model', required=True,
                        help='name of the requested google ai model')
    parser.add_argument('--annotations', required=True,
                        help='path of the annotations folder')
    parser.add_argument('--output', required=True,
                        help='path of the output folder')
    args = parser.parse_args()

    print(f"Loading {args.model}.")
    accelerator = Accelerator()
    model = AutoModelForCausalLM.from_pretrained(argparse.model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(argparse.model)
    model = accelerator.prepare(model)
    print(f"Loaded.")

    print(f"Requesting {args.model}.")
    process_model_requests(args.annotations, args.output, model)
    print(f"Requested.")