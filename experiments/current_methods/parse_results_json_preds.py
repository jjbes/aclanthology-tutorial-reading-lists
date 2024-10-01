import os
import re
import json
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup
import typing_extensions as typing 

def clean_title(title: str) -> str:
    title = re.sub('["*]', '', title)  # Remove quotes and asterisks
    return title.strip('"').strip('*').strip('"')

def extract_title_from_quotes(title: str) -> str:
    quoted_title_match = re.search(r'"(.+?)"', title)
    quoted_title_trailing_author_match = re.search(r'(.+?) by', title)
    bold_title_match = re.search(r'\*\*(.+?)\*\*', title)
    trailing_bold_match = re.search(r'([^:]+?)\*\*', title)

    if quoted_title_match:
        return quoted_title_match.group(1)
    elif quoted_title_trailing_author_match: 
        return quoted_title_trailing_author_match.group(1)
    elif bold_title_match:
        return bold_title_match.group(1)
    elif trailing_bold_match:
        return trailing_bold_match.group(1)
   
    return title  # Return original if no matches found

def parse_year(date: typing.Optional[str]) -> typing.Optional[int]:
    if date and re.match(r"^\d{4}$", date):
        return int(date)
    return None

def parse_extracted_json(path: str) -> typing.Generator[typing.Dict[str, typing.Optional[str]]]:
    with open(path, "r") as file:  
        content = json.load(file)
        for item in content:
            if "citation-number" in item and "title" in item:
                title = item["title"][0].replace("\\", "")
                title = extract_title_from_quotes(title)
                title = clean_title(title)
                year = parse_year(item.get("date", [None])[0])
                yield {"title": title, "year": year}

def parse_json(path:str) -> list[typing.Dict[str, typing.Optional[str]]]:
    with open(str(path), "r") as file:      
        try:
            data = json.load(file)
            return [{"title": item.get("title"), "year": item.get("year")} for item in data]
        except ValueError:
            return []

def find_articles(html:str) -> str:
    soup = BeautifulSoup(html, 'html.parser') 
    results = []
    for refs in soup.find_all("div", {"class": "gs_r gs_or gs_scl"}):
        if refs.find("h3").find("a"):
            title = refs.find("h3").find("a").text
        else:
            title = refs.find("h3").find_all('span')[3].text #Citations
        year = refs.find("div", {"class": "gs_a"}).text.split(" - ")[0][-4:]
        results.append({
            "title": title,
            "year": year
        })
    return results

def parse_html(pathlist:list[str]) -> typing.Dict:
    preds = {}
    for path in pathlist:
        key = path.parts[-1].replace(".html", "")
        with open(str(path)) as file:
            preds[key] = find_articles(file)
    return preds         

def process_parse_results(results_name:str, model_type:str, parse_func:typing.Callable) -> None:
    for annotator_num in [1, 2, 3]:
        # Save file location
        folder_path = f"{model_type}/preds/{results_name}"
        os.makedirs(folder_path, exist_ok=True)
        file_path = f"{folder_path}/preds_annot{annotator_num}.json"
        # Load annotator queries from CSV
        if not os.path.exists(file_path):
            results_paths = sorted(Path(f"{model_type}/results/{results_name}/annotator{annotator_num}/").glob(f'*.json'))
            results = {path.parts[-1].replace(f".json", ""): list(parse_func(path)) for path in tqdm(results_paths, desc=f"{results_name} (A{annotator_num})")}
            with open(file_path, "w") as file:
                json.dump(results , file) 

if __name__ == "__main__":
    process_parse_results("gpt-4o", "instructs_models", parse_extracted_json)
    process_parse_results("gpt-4o_json", "instructs_models", parse_json)
    process_parse_results("gpt-4o-2024-08-06", "instructs_models", parse_extracted_json)
    process_parse_results("gpt-4o-2024-08-06_json", "instructs_models", parse_json)
    process_parse_results("gpt-4o-2024-08-06_structured_output", "instructs_models", parse_json)
    process_parse_results("gemini-1.5-flash", "instructs_models", parse_extracted_json)
    process_parse_results("gemini-1.5-flash_json", "instructs_models", parse_json)
    process_parse_results("google_scholar", "search_engines", parse_html)