import json
import os
from tqdm.asyncio import tqdm_asyncio
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

import asyncio
import aiohttp
from collections import defaultdict

REQUESTS_PER_MINUTE = 100
CONCURRENT_REQUESTS = 10
RATE_LIMIT_WINDOW = 60.0 / REQUESTS_PER_MINUTE

async def fetch_s2(query, session, semaphore):
    async with semaphore:
        async with session.get(
                                'https://api.semanticscholar.org/graph/v1/paper/search/match', 
                                params={"query": query}, 
                                headers={"x-api-key": os.environ['SEMANTIC_SCHOLAR_API']}
                            ) as response:
            if response.status == 200:
                data = await response.json()
                await asyncio.sleep(RATE_LIMIT_WINDOW)
                return data["data"][0]["paperId"] if "data" in data and len(data["data"]) > 0 else None
 
async def match_s2_id(preds):
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS) 
    matched = defaultdict(list)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for k, vs in preds.items():
            if len(vs):
                for v in vs:
                    task = fetch_s2(v["title"], session, semaphore)
                    tasks.append((k, task, v))
            else:
                matched[k] = []

        results = await tqdm_asyncio.gather(*[task for _, task, _ in tasks])
        for i, (k, task, v) in enumerate(tasks):
            scholar_id = results[i]
            matched[k].append({
                "id": scholar_id,
                "title": v["title"],
                "year": v.get("year", None)
            })
    return matched

#Scholarly Engines
for model in ["google_scholar"]:
    for annotator_i in [1,2,3]:
        preds = json.loads(Path(f'search_engines/preds/{model}/preds_annot{annotator_i}.json').read_text())
        preds_ids = asyncio.run(match_s2_id(preds))
        with open(f'preds/{model}/preds_annot{annotator_i}.json', 'w') as f:
            json.dump(preds_ids, f)

#Instruction models
for model in ["gpt-4o","gpt-4o_json", "gpt-4o-2024-08-06", "gemini-1.5-flash"]:
    for annotator_i in [1,2,3]:
        preds = json.loads(Path(f'instruct_models/preds/{model}/preds_annot{annotator_i}.json').read_text())
        preds_ids = asyncio.run(match_s2_id(preds))
        with open(f'preds/{model}/preds_annot{annotator_i}.json', 'w') as f:
            json.dump(preds_ids, f)