from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import json
from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import faiss
import os
from pathlib import Path

def _generate_embeddings(batch):
    inputs = tokenizer(batch, padding=True, truncation=True,
        return_tensors="pt", return_token_type_ids=False, max_length=512)
    inputs.to(device)
    with torch.no_grad():
        output = model(**inputs)
        embeddings = output.last_hidden_state[:, 0, :]
        return np.array(embeddings.cpu(), np.float32) 
    
def _batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
def _embed_content(content, batch_size=64):
    for batch in _batch(content, batch_size):
        yield _generate_embeddings(batch)
def embed_content(content, batch_size=64): return np.concatenate(list(_embed_content(content, batch_size=batch_size)))

def embed_dict_and_store(index, dict_):
    embeddings = embed_content(list(dict_.values()))
    index.add(embeddings)

def corpus_table_matching(dict_):
    return {i: k for k, i in zip(dict_.keys(), range(0, len(dict_)))}

def generate_list(queries):
    preds_embeddings = defaultdict(list)
    for k, query in queries.items():
        if query["query_keywords"]:
            _, I = acl_anthology_faiss_index.search(embed_content([query["query_keywords"]]), 97440) 
    
            i=0
            while len(preds_embeddings[k])<20:
                if acl_anthology_dataset[embedding_keys_index[I[0][i]]]["year"] <= query["year"]:
                    preds_embeddings[k].append({
                        "title": acl_anthology_dataset[embedding_keys_index[I[0][i]]]["title"],
                        "year": acl_anthology_dataset[embedding_keys_index[I[0][i]]]["year"]
                    })
                i=i+1
        else:
            preds_embeddings[k] = []
    return preds_embeddings

device = "mps" #'cuda' for Nvidia GPU / 'mps' for Apple M-series
tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
model.to(device)
model.eval()

with open('acl_anthology_dataset/acl_anthology_dataset.json', 'r') as file: 
    acl_anthology_dataset = json.load(file) 
documents = { k: tokenizer.sep_token.join(filter(None, [val['title'], (val['abstract'] if val['abstract'] else None)])) for k, val in acl_anthology_dataset.items()}
# Uncomment below to regenerate index
#acl_anthology_faiss_index = faiss.IndexFlatIP(768)
#embed_dict_and_store(acl_anthology_faiss_index, documents)
#faiss.write_index(acl_anthology_faiss_index, "acl_anthology_dataset/acl_anthology_faiss_index_cosine")
acl_anthology_faiss_index = faiss.read_index("acl_anthology_dataset/acl_anthology_faiss_index_cosine")
embedding_keys_index = {i: k for k, i in zip(documents.keys(), range(acl_anthology_faiss_index.ntotal))}

MODEL_NAME = "specterv2"
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