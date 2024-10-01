import os
import json
import dotenv
import asyncio
import aiohttp
from pathlib import Path
import typing_extensions as typing 
from collections import defaultdict
from tqdm.asyncio import tqdm_asyncio
from asynciolimiter import StrictLimiter

dotenv.load_dotenv()

# 500 RPM
RATE_LIMITER = StrictLimiter(499/60)

async def get_id(query:str, session:typing.Any) -> typing.Optional[str]:

    await RATE_LIMITER.wait()
    async with session.get(
                            'https://api.semanticscholar.org/graph/v1/paper/search/match', 
                            params={"query": query}, 
                            headers={"x-api-key": os.environ['SEMANTIC_SCHOLAR_API']}
                        ) as response:
        if response.status == 200:
            data = await response.json()
            return data["data"][0]["paperId"] if "data" in data and len(data["data"]) > 0 else None
 
async def fetch_s2_batch(preds_list:list[typing.Dict], desc:typing.Optional[str]=None) -> list[typing.Dict] :
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

def process_s2_match(preds_name:str, model_type:str) -> None:
    for annotator_num in [1, 2, 3]:
        file_path = f'{model_type}/preds/{preds_name}/preds_annot{annotator_num}.json'
        preds = json.loads(Path(file_path).read_text())
        preds_ids = asyncio.run(fetch_s2_batch(preds, desc=f"{preds_name} (A{annotator_num})"))
        with open(file_path, 'w') as f:
            json.dump(preds_ids, f)

if __name__ == "__main__":
    process_s2_match("google_scholar", "search_engines")
    process_s2_match("gpt-4o", "instructs_models")
    process_s2_match("gpt-4o_json", "instructs_models")
    process_s2_match("gpt-4o-2024-08-06", "instructs_models")
    process_s2_match("gpt-4o-2024-08-06_json", "instructs_models")
    process_s2_match("gpt-4o-2024-08-06_structured_output", "instructs_models")
    process_s2_match("gemini-1.5-flash", "instructs_models")
    process_s2_match("gemini-1.5-flash_json", "instructs_models")