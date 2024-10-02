import os
import dotenv
import numpy as np
import pandas as pd  
from typing import List, Dict, Optional, Callable, TypedDict
import asyncio
from tqdm.asyncio import tqdm_asyncio
from asynciolimiter import StrictLimiter
import google.generativeai as genai

dotenv.load_dotenv()
genai.configure()

""" Async basic generation from Google GenAI API """
async def generate_content_base(model_name:str, query: str, max_year: int) -> str:
    await RATE_LIMITER.wait()
    query = query.replace("Give me a reading list", f"Give me a reading list of 20 articles up to {max_year}")
    
    generation_config = genai.GenerationConfig(
        temperature=0
    )
    model = genai.GenerativeModel(model_name)
    try:
        response = await asyncio.to_thread(model.generate_content, query, generation_config=generation_config)
        return response.text
    except Exception as e:
        print(f"Exception: {e}")
        return None
    

""" Async JSON-Mode generation from Google GenAI API """
async def generate_content_json(model_name:str, query: str, max_year: int) -> str:
    await RATE_LIMITER.wait()
    query = query.replace("Give me a reading list", f"Give me a reading list of 20 articles up to {max_year}")
    
    class Article(TypedDict):
        title: str
        authors: List[str]
        year: int

    generation_config = genai.GenerationConfig(
        response_mime_type="application/json", 
        response_schema=List[Article], 
        temperature=0
    )
    model = genai.GenerativeModel(model_name)
    try:
        response = await asyncio.to_thread(model.generate_content, query, generation_config=generation_config)
        return response.text
    except Exception as e:
        print(f"Exception: {e}")
        return None

""" Process a batch of queries asynchronously """
async def fetch_completions_batch(
        model_name:str, 
        queries:List[Dict], 
        generation_func:Callable, 
        desc:Optional[str]=None
    ) -> Dict:

    tasks = [
        (query["id"], asyncio.create_task(generation_func(model_name, query["query_sentence"], query["year"]))) 
        for query in queries
    ]
    results = await tqdm_asyncio.gather(*[task for _, task in tasks], desc=desc)
    return {k: results[i] for i, (k, _) in enumerate(tasks) if results[i]}

""" Process requests for a specific model and save results """
def process_model_requests(
        model_name:str, 
        generation_func:Callable, 
        folder:str = "", 
        format:str = "json"
    ) -> None:

    folder = folder or model_name
    for annotator_num in [1, 2, 3]:
        # Load annotator queries from CSV
        query_file = f"../../../annotations/annotation_{annotator_num}.csv"
        folder_path = f"results/{folder}/annotator{annotator_num}/"
        annotator_queries = pd.read_csv(query_file)[["id", "year", "query_sentence"]].replace(np.nan, None).to_dict(orient='records')
        annotator_queries = [
            query for query in annotator_queries 
            #Filter out already generated files
            if not os.path.exists(f"{folder_path}/{query['id']}.{format}")
        ]
        # Fetch completions for the batch
        predictions = asyncio.run(fetch_completions_batch(model_name, annotator_queries, generation_func, desc=f"{model_name} (A{annotator_num})"))
        # Save each result as a markdown file
        for query_id, content in predictions.items():
            os.makedirs(folder_path, exist_ok=True)
            with open(f"{folder_path}/{query_id}.{format}", "w") as file:
                file.write(content) 

if __name__ == "__main__":
    # 15 RPM
    RATE_LIMITER = StrictLimiter(15/60)
    process_model_requests("gemini-1.5-flash", generate_content_base, format="md")
    process_model_requests("gemini-1.5-flash", generate_content_json, folder="gemini-1.5-flash_json", format="json")

    # 2 RPM
    RATE_LIMITER = StrictLimiter(2/60)
    process_model_requests("gemini-1.5-pro", generate_content_json, folder="gemini-1.5-pro_json", format="json")
