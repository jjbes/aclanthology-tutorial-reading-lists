import os
import argparse
from tqdm import tqdm
from pathlib import Path

""" Extract references as .json from .md files using anystyle """
def extract_json(folder:str) -> None:
    for annotator_num in [1, 2, 3]:
        files = sorted(Path(f"{folder}/annotator{annotator_num}/").glob(f'*.md'))
        for path in tqdm(files, desc=f"A{annotator_num}"):
            #[sudo] gem install anystyle-cli
            os.system(f"anystyle -f json parse {str(path)} > {str(path).replace('.md', '.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract references from markdown files using Anystyle')
    parser.add_argument('--folder', required=True,
                        help='path of the folder to process')
    args = parser.parse_args()

    print(f"Extracting references from Markdown to JSON files. ({args.folder})")
    extract_json(args.folder)
    print(f"Extracted.")