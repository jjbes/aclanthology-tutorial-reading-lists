import os
import json
import dotenv
import argparse
import numpy as np
import pandas as pd   
from typing import Callable, Optional
from pydantic import BaseModel
import openai
import asyncio
from tqdm.asyncio import tqdm_asyncio
from asynciolimiter import StrictLimiter

dotenv.load_dotenv()

""" Async basic completion from the OpenAI API """
async def get_completion_base(model_name:str, query:str, max_year:int) -> str:
    await RATE_LIMITER.wait()
    async with openai.AsyncOpenAI() as client:
        query = query.replace("Give me a reading list", f"Give me a reading list of 20 articles up to {max_year}")
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": query
                }
                ]
            }
            ],
            temperature=0,
            max_tokens=4095,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

    return response.choices[0].message.content

""" Async JSON-Mode completion from the OpenAI API """
async def get_completion_json(model_name:str, query:str, max_year:int) -> str:
    await RATE_LIMITER.wait()
    async with openai.AsyncOpenAI() as client:
        query = query.replace("Give me a reading list", f"Give me a reading list of 20 articles up to {max_year}")
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": query
                }
                ]
            }
            ],
            functions = [
                {
                "name": "generate_reading_list",
                "description": "Generate a reading list following the Article and ReadingList schema.",
                "parameters": {
                    "type": "object",
                    "properties": {
                    "reading_list": {
                        "type": "array",
                        "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                            "type": "string",
                            "description": "The title of the article."
                            },
                            "authors": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of authors for the article."
                            },
                            "year": {
                            "type": "integer",
                            "description": "Year the article was published."
                            }
                        },
                        "required": ["title", "authors", "year"]
                        }
                    }
                    },
                    "required": ["reading_list"]
                }
                }
            ],
            function_call= { "name": "generate_reading_list" },
            temperature=0,
            max_tokens=4095,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return json.dumps(json.loads(response.choices[0].message.function_call.arguments)["reading_list"])

""" Async Structured Output completion from the OpenAI API  """
async def get_completion_structured_output(model_name:str, query:str, max_year:int) -> str:
    await RATE_LIMITER.wait()

    class Article(BaseModel):
        title: str
        authors: list[str]
        year: int

    class ReadingList(BaseModel):
        reading_list: list[Article]

    async with openai.AsyncOpenAI() as client:
        query = query.replace("Give me a reading list", f"Give me a reading list of 20 articles up to {max_year}")
        response = await client.beta.chat.completions.parse(
            model=model_name,
            messages=[
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": query
                }
                ]
            }
            ],
            temperature=0,
            max_tokens=4095,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format=ReadingList
        )
    return json.dumps(response.choices[0].message.parsed.model_dump()["reading_list"])

""" Process a batch of queries asynchronously """
async def fetch_completions_batch(
        model_name:str, 
        queries:str, 
        completion_func:Callable, 
        desc:Optional[str]=None
    ) -> dict:

    tasks = [
        (query["id"], asyncio.create_task(completion_func(model_name, query["query_sentence"], query["year"]))) 
        for query in queries
    ]
    results = await tqdm_asyncio.gather(*[task for _, task in tasks], desc=desc)
    return {k: results[i] for i, (k, _) in enumerate(tasks)}

""" Process requests for a specific model and save results """
def process_model_requests(
        model_name:str, 
        completion_func:Callable, 
        annotation_path:str = "", 
        output_path:str = "", 
        format:str = "json"
    ) -> None:
    output_path = output_path or model_name
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
        predictions = asyncio.run(fetch_completions_batch(model_name, annotator_queries, completion_func, desc=f"A{annotator_num}"))
        # Save each result as a markdown file
        for query_id, content in predictions.items():
            os.makedirs(folder_path, exist_ok=True)
            with open(f"{folder_path}/{query_id}.{format}", "w") as file:
                file.write(content) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Request OpenAI API for the generation of a reading list')
    parser.add_argument('--model', required=True,
                        help='name of the requested google ai model')
    parser.add_argument('--annotations', required=True,
                        help='path of the annotations folder')
    parser.add_argument('--output', required=True,
                        help='path of the output folder')
    parser.add_argument('--output_type', required=True,
                        help='type of generation for the outputs',
                        choices=["base","json_mode", "structured_output"])
    parser.add_argument('--rate_limit', required=True, type=int,
                        help='rate limit of the API (RPM)')
    args = parser.parse_args()
       
    if args.output_type == "json_mode":
        completion_function = get_completion_json
        format = "json"
    elif args.output_type == "structured_output":
        completion_function = get_completion_structured_output
        format = "json"
    else:
        completion_function = get_completion_base
        format = "md"

    RATE_LIMITER = StrictLimiter(args.rate_limit/60)
    print(f"Requesting {args.model}.")
    process_model_requests(args.model, completion_function, format=format, annotation_path=args.annotations, output_path=args.output)
    print(f"Requested.")