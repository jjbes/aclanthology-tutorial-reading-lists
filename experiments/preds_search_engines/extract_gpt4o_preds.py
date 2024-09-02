import re
import json
from tqdm import tqdm
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

def extract_refs(path):
    #[sudo] gem install anystyle-cli
    os.system(f"anystyle -f json parse {str(path)} > {str(path).replace('.md', '.json')}")

for annotator_i in [1,2,3]:
    for run_i in [1,2,3]:
        files = sorted(Path(f"search_engine_results/gpt-4o/annotator{annotator_i}/run{run_i}/").glob(f'*.md'))
        for path in tqdm(files):
            extract_refs(path)