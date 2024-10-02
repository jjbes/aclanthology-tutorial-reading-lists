import os
import json
import bm25s
import Stemmer
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

""" Retrieve BM25 top 20 relevant documents"""
def retrieve_relevant(
        query:str, 
        year:str, 
        retriever:bm25s.BM25, 
        stemmer:Stemmer, 
        inverted_documents:dict
    ) -> list:
    reading_list = []
    if query:
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer)
        docs, _ = retriever.retrieve(query_tokens, k=len(acl_anthology_dataset))

        for doc in docs[0]:
            key = inverted_documents[doc]
            doc = acl_anthology_dataset[key]

            if doc["year"] <= year:
                reading_list.append({
                    "title": doc["title"],
                    "year": doc["year"]
                })

            if len(reading_list) >= 20:
                break
    return reading_list

""" Proccess BM25 for each annotation"""
def process_retrieval(
        annotations_path:str,
        output_path:str,
        retriever:bm25s.BM25, 
        stemmer:Stemmer, 
        inverted_documents:dict
    ) -> None:

    for annotator_num in [1, 2, 3]:
        annotations = pd.read_csv(f"{annotations_path}/annotation_{annotator_num}.csv")[["id", "year", "query_keywords"]].replace(np.nan, None).to_dict(orient='records')
        predictions = {
            annotation["id"]: retrieve_relevant(
                annotation["query_keywords"],
                annotation["year"],
                retriever,
                stemmer,
                inverted_documents
            ) 
            for annotation in tqdm(annotations, desc=f"A{annotator_num}")
        }
        os.makedirs(output_path, exist_ok=True)
        with open(f"{output_path}/preds_annot{annotator_num}.json", "w") as file:
            json.dump(predictions, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use BM25 for the generation of a reading list')
    parser.add_argument('--dataset', required=True,
                        help='path of the dataset folder')
    parser.add_argument('--annotations', required=True,
                        help='path of the annotations folder')
    parser.add_argument('--output', required=True,
                        help='path of the output folder')
    args = parser.parse_args()

    with open(f'{args.dataset}/acl_anthology_dataset.json', 'r') as file: 
        acl_anthology_dataset = json.load(file) 

    documents = { 
        k: ' '.join(filter(None, [val['title'], (val['abstract'] if val['abstract'] else None)])) 
        for k, val in acl_anthology_dataset.items()
    }
    inverted_documents = {v: k for k,v in documents.items()}

    print(f"Generating BM25 index.")
    stemmer = Stemmer.Stemmer("english")
    corpus = list(documents.values())
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(corpus_tokens)
    print(f"Generated.")
    
    print(f"Predicting BM25 most similar documents.")   
    process_retrieval(args.annotations, args.output, retriever, stemmer, inverted_documents)
    print(f"Predicted.")   