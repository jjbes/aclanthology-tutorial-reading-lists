import json
import os
import requests
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

from functools import partial
tqdm = partial(tqdm, position=0, leave=True)

def match_semantic_scholar_id_request(query):   
    apiKey = os.environ['SEMANTIC_SCHOLAR_API']
    payload = {
        "query":query,
    }
    r = requests.get('https://api.semanticscholar.org/graph/v1/paper/search/match', params=payload, headers={"x-api-key":apiKey})
    return r.json()["data"][0]["paperId"] if "data" in r.json() else None

def match_semantic_scholar_id(preds):
    matched = defaultdict(list)
    for k,vs in tqdm(preds.items()):
        if len(vs):
            for v in vs:
                matched[k].append({
                    "id": match_semantic_scholar_id_request(v["title"]),
                    "title":v["title"],
                    "year":v["year"] if "year" in v else None
                })
        else:
            matched[k] = []
    return matched

# Google Scholar
for annotator_i in [1,2,3]:
    preds_google_scholar_annot_ids = match_semantic_scholar_id(json.loads(Path(f'preds/google_scholar/preds_annot{annotator_i}.json').read_text()))
    with open(f'preds/google_scholar/preds_annot{annotator_i}.json', 'w') as f:
        json.dump(preds_google_scholar_annot_ids, f)

# GPT-4o
for annotator_i in [1,2,3]:
    for run_i in [1,2,3]:
        preds_gpt4o_annot_ids = match_semantic_scholar_id(json.loads(Path(f'preds/gpt-4o/run{run_i}/preds_annot{annotator_i}.json').read_text()))
        with open(f'preds/gpt-4o/run{run_i}/preds_annot{annotator_i}.json', 'w') as f:
            json.dump(preds_gpt4o_annot_ids, f)