import os
import json
import bm25s
import Stemmer
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

def generate_list(queries):
    preds = defaultdict(list)
    for id_, query in queries.items():
        if query["query_keywords"]:
            query_tokens = bm25s.tokenize(query["query_keywords"], stopwords="en", stemmer=stemmer)
            docs, _ = retriever.retrieve(query_tokens, k=len(acl_anthology_dataset))
            reading_list = []
            i = 0
            while len(reading_list) < 20:
                key = inverted_documents[docs[0, i]]
                if acl_anthology_dataset[key]["year"] <= query["year"]:
                    reading_list.append({
                        "title": acl_anthology_dataset[key]["title"],
                        "year": acl_anthology_dataset[key]["year"]
                    })
                i+=1
            preds[id_] = reading_list
        else:
            preds[id_] = []
    return preds

with open('acl_anthology_dataset/acl_anthology_dataset.json', 'r') as file: 
    acl_anthology_dataset = json.load(file) 
documents = { k: ' '.join(filter(None, [val['title'], (val['abstract'] if val['abstract'] else None)])) for k, val in acl_anthology_dataset.items()}
inverted_documents = {v: k for k,v in documents.items()}


if __name__ == "__main__":
    stemmer = Stemmer.Stemmer("english")

    corpus = list(documents.values())
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(corpus_tokens)

    MODEL_NAME = "bm25"
    for annotator_i in [1,2,3]:
        print(f"Requesting annotator {annotator_i}")
        annotator_queries = pd.read_csv(f"../../annotations/annotation_{annotator_i}.csv")[["id", "year", "query_keywords"]].replace(np.nan, None).to_dict(orient='records')
        annotator_queries = {query["id"]: query for query in annotator_queries}
        for query in tqdm(annotator_queries):
            FOLDER_PATH = f"preds/{MODEL_NAME}/"
            if not os.path.exists(FOLDER_PATH):
                os.makedirs(FOLDER_PATH)
            FILE_PATH = f"{FOLDER_PATH}/preds_annot{annotator_i}.json"
            if not Path(FILE_PATH).is_file():
                preds = generate_list(annotator_queries)
                with open(FILE_PATH, "w") as f:
                    json.dump(preds, f) 