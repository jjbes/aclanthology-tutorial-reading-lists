from tqdm import tqdm
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

def extract(path):
    #[sudo] gem install anystyle-cli
    os.system(f"anystyle -f json parse {str(path)} > {str(path).replace('.md', '.json')}")

MODEL_NAME = "gpt-4o"
for annotator_i in [1,2,3]:
    files = sorted(Path(f"results/{MODEL_NAME}/annotator{annotator_i}/").glob(f'*.md'))
    for path in tqdm(files):
        extract(path)