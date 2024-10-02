import os
import dotenv
import argparse
import numpy as np
import pandas as pd  
from typing import Optional, Callable
from typing_extensions import TypedDict
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
        authors: list[str]
        year: int

    generation_config = genai.GenerationConfig(
        response_mime_type="application/json", 
        response_schema=list[Article], 
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
        queries:list[dict], 
        generation_func:Callable, 
        desc:Optional[str]=None
    ) -> dict:

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
        annotation_path:str = "", 
        output_path:str = "", 
        format:str = "json"
    ) -> None:

    for annotator_num in [1, 2, 3]:
        # Load annotator queries from CSV
        query_file = f"{annotation_path}/annotation_{annotator_num}.csv"
        folder_path = f"{output_path}/annotator{annotator_num}/"
        annotator_queries = pd.read_csv(query_file)[["id", "year", "query_sentence"]].replace(np.nan, None).to_dict(orient='records')
        annotator_queries = [
            query for query in annotator_queries 
            #Filter out already generated files
            if not os.path.exists(f"{folder_path}/{query['id']}.{format}")
        ]
        # Fetch completions for the batch
        predictions = asyncio.run(fetch_completions_batch(model_name, annotator_queries, generation_func, desc=f"A{annotator_num}"))
        # Save each result as a markdown file
        for query_id, content in predictions.items():
            os.makedirs(folder_path, exist_ok=True)
            with open(f"{folder_path}/{query_id}.{format}", "w") as file:
                file.write(content) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Request Google AI API for the generation of a reading list')
    parser.add_argument('--model', required=True,
                        help='name of the requested google ai model')
    parser.add_argument('--annotations', required=True,
                        help='path of the annotations folder')
    parser.add_argument('--output', required=True,
                        help='path of the output folder')
    parser.add_argument('--output_type', required=True,
                        help='type of generation for the outputs',
                        choices=["base","json_mode"])
    parser.add_argument('--rate_limit', required=True, type=int,
                        help='rate limit of the API (RPM)')
    args = parser.parse_args()
       
    if args.output_type == "json_mode":
        completion_function = generate_content_json
        format = "json"
    else:
        completion_function = generate_content_base
        format = "md"

    RATE_LIMITER = StrictLimiter(args.rate_limit/60)
    print(f"Requesting {args.model}.")
    process_model_requests(args.model, completion_function, format=format, annotation_path=args.annotations, output_path=args.output)
    print(f"Requested.")
