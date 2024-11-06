import os
import json
import dotenv
import asyncio
import aiohttp
import argparse
from pathlib import Path
from typing import Optional
from collections import defaultdict
from tqdm.asyncio import tqdm_asyncio
from asynciolimiter import StrictLimiter

dotenv.load_dotenv()

# 500 RPM
RATE_LIMITER = StrictLimiter(499/60)

"""  Async request ids from Semantic Scholar API """
async def get_id(query:str, session:aiohttp.ClientSession) -> Optional[str]:
    await RATE_LIMITER.wait()
    async with session.get(
                            'https://api.semanticscholar.org/graph/v1/paper/search/match', 
                            params={
                                "query": query,
                                "fields":"externalIds",
                            }, 
                            headers={"x-api-key": os.environ['SEMANTIC_SCHOLAR_API']}
                        ) as response:
        if response.status == 200:
            data = await response.json()
            return data["data"][0]["externalIds"]["CorpusId"] if "data" in data and len(data["data"]) > 0 else None
 
"""  Fetch Semantic Scholar API """
async def fetch_s2_batch(preds_list:list[dict], desc:Optional[str]=None) -> list[dict] :
    matched = defaultdict(list)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for k, preds in preds_list.items():
            if len(preds):
                for pred in preds:
                    task = get_id((pred.get("title") or ''), session)
                    tasks.append((k, task, pred))
            else:
                matched[k] = []
        results = await tqdm_asyncio.gather(*[task for _, task, _ in tasks], desc=desc)
        for i, (k, task, pred) in enumerate(tasks):
            scholar_id = results[i]
            matched[k].append({
                "id": scholar_id,
                "title": pred["title"],
                "year": pred.get("year")
            })
    return matched

"""  Matching predicted articles to an existing Semantic Scholar's ids """
def process_s2_match(folder_path:str) -> None:
    for annotator_num in [1, 2, 3]:
        file_path = f'{folder_path}/preds_annot{annotator_num}.json'
        preds = json.loads(Path(file_path).read_text())
        preds_ids = asyncio.run(fetch_s2_batch(preds, desc=f"A{annotator_num}"))
        with open(file_path, 'w') as f:
            json.dump(preds_ids, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse results of a models to a predictions file')
    parser.add_argument('--folder', required=True,
                        help='path of the folder to process')
    args = parser.parse_args()

    print(f"Matching predictions titles to Semantic Scholar. ({args.folder})")
    process_s2_match(args.folder)
    print(f"Matched.")