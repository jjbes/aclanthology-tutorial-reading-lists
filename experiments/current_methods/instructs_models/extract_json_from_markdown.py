from tqdm import tqdm
import os
from pathlib import Path

def extract_json(results_name:str) -> None:
    print(f"Extracting JSON from results of model: {results_name}")
    for annotator_num in [1, 2, 3]:
        files = sorted(Path(f"results/{results_name}/annotator{annotator_num}/").glob(f'*.md'))
        for path in tqdm(files):
            #[sudo] gem install anystyle-cli
            os.system(f"anystyle -f json parse {str(path)} > {str(path).replace('.md', '.json')}")

if __name__ == "__main__":
    extract_json("gpt-4o")
    extract_json("gpt-4o-2024-08-06")
    extract_json("gemini-1.5-flash")