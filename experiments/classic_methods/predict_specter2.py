import os
import json
import torch
import faiss
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from adapters import AutoAdapterModel
from transformers import AutoTokenizer
from typing import Generator

device = "mps" #'cuda' for Nvidia GPU / 'mps' for Apple M-series

""" Load and initialize the tokenizer and model with the adapter """
def load_model_and_tokenizer() -> tuple:
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
    model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
    model.to(device)
    model.eval()
    return tokenizer, model

""" Prepare document for tokenization """
def prepare_documents(dataset:dict, tokenizer:AutoTokenizer) -> dict:
    return {
        k: tokenizer.sep_token.join(filter(None, [val['title'], val.get('abstract')])) 
        for k, val in dataset.items()
    }

""" Generate embeddings for a batch of text """
def generate_embeddings(batch:list) -> np.ndarray:
    with torch.no_grad():
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        embeddings = model(**inputs).last_hidden_state[:, 0, :]
    return embeddings.cpu().numpy().astype(np.float32)
    
""" Batch chunks of size n """
def batch(iterable:list, n:int=1) -> Generator:
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

"""Embed content with batching """
def embed_content(content:list, batch_size:int=64) -> np.ndarray:
    return np.concatenate([generate_embeddings(list(batch)) for batch in batch(content, batch_size)])

""" Compute similarity of query to an index by maximum year"""
def compute_similarity(
        query:str, 
        year:int,
        index: faiss.Index, 
        index_keys:dict, 
        dataset:dict
    ) -> list:
    reading_list = []
    if query:
        _, I = index.search(embed_content([query]), 97440) 
        for doc in I[0]:
            doc = dataset[index_keys[doc]]

            if doc["year"] <= year:
                reading_list.append({
                    "title": doc["title"],
                    "year": doc["year"]
                })

            if len(reading_list) >= 20:
                break
    return reading_list

""" Proccess BM25 for each annotation """
def process_retrieval(
        annotations_path:str,
        output_path:str,
        index: faiss.Index, 
        index_keys:dict, 
        dataset:dict
    ) -> None:

    for annotator_num in [1, 2, 3]:
        annotations = pd.read_csv(f"{annotations_path}/annotation_{annotator_num}.csv")[["id", "year", "query_keywords"]].replace(np.nan, None).to_dict(orient='records')
        predictions = {
            annotation["id"]: compute_similarity(
                annotation["query_keywords"],
                annotation["year"],
                index, 
                index_keys, 
                dataset
            ) 
            for annotation in tqdm(annotations, desc=f"A{annotator_num}")
        }
        os.makedirs(output_path, exist_ok=True)
        with open(f"{output_path}/preds_annot{annotator_num}.json", "w") as file:
            json.dump(predictions, file)

""" Create a mapping of index positions to document keys."""
def create_index_keys(documents:dict, index:faiss.Index) -> dict:
    return {i: k for k, i in zip(documents.keys(), range(index.ntotal))}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use SPECTER2 for the generation of a reading list')
    parser.add_argument('--dataset', required=True,
                        help='path of the dataset folder')
    parser.add_argument('--annotations', required=True,
                        help='path of the annotations folder')
    parser.add_argument('--output', required=True,
                        help='path of the output folder')
    args = parser.parse_args()

    tokenizer, model = load_model_and_tokenizer()

    with open(f'{args.dataset}/acl_anthology_dataset.json', 'r') as file: 
        dataset = json.load(file) 

    documents = prepare_documents(dataset, tokenizer)

    # Uncomment below to regenerate index
    # print(f"Generating SPECTER2 index.")
    # index = faiss.IndexFlatIP(768)
    # embeddings = embed_content(list(documents.values()))
    # index.add(embeddings)
    # faiss.write_index(index, f"{args.dataset}/acl_anthology_faiss_index_cosine")
    # print(f"Generated.")

    index = faiss.read_index(f"{args.dataset}/acl_anthology_faiss_index_cosine")

    index_keys = create_index_keys(documents, index)

    print(f"Predicting SPECTER2 most similar documents.")
    process_retrieval(args.annotations, args.output, index, index_keys, dataset)
    print(f"Predicted.")