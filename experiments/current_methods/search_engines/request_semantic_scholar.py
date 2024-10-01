import os
import json
import dotenv
import requests
import pandas as pd
from tqdm import tqdm
import typing_extensions as typing 

dotenv.load_dotenv()

# S2 API rate limit is 1RPM, async is not needed

def get_papers(query:str, max_year:typing.Optional[int]=None) -> typing.Dict:   
    apiKey = os.environ['SEMANTIC_SCHOLAR_API']
    payload = {
        "query":query,
        "offset":0,
        "limit":20,
        "fields":"title,year",
        "year": f"-{max_year}" if max_year else "",
    }
    r = requests.get('https://api.semanticscholar.org/graph/v1/paper/search', params=payload, headers={"x-api-key":apiKey})
    return r.json()

def fetch_s2(query:str, max_year:typing.Optional[int]=None) -> list:
    results = []
    response = get_papers(query, max_year=max_year)
    if "data" in response :
        for paper in response["data"]:
            results.append({
                "id": paper["paperId"],
                "title": paper["title"],
                "year": paper["year"]
            })
    return results     

def process_s2_request() -> None :
    for annotator_num in [1, 2, 3]:
        annotator_queries = pd.read_csv(f"../../../annotations/annotation_{annotator_num}.csv")[["id", "year", "query_keywords"]].to_dict(orient='records')
        preds_annot = {query["id"]: fetch_s2(query["query_keywords"], max_year=query["year"]) for query in tqdm(annotator_queries, desc=f"Semantic Scholar (A{annotator_num})")}
        with open(f"preds/semantic_scholar/preds_annot{annotator_num}.json", "w") as fp:
            json.dump(preds_annot , fp) 

if __name__ == "__main__":
    process_s2_request()