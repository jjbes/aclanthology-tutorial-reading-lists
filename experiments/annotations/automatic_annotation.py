import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("bart-large-kp20k", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("bart-large-kp20k", local_files_only=True)


def embed(text):
    input_ids = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**input_ids, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).replace(";",',')

reading_lists = pd.read_csv("../../reading_lists.csv")
reading_lists = reading_lists.replace(np.nan, None)

bart_queries = [{"id":id, "title":title, "abstract":abstract, "year":year, "query_keywords":embed(title+"<s>"+(abstract or ""))} for id, title, abstract, year in tqdm(zip(reading_lists["id"], reading_lists["title"], reading_lists["abstract"], reading_lists["year"]))]
df = pd.DataFrame(bart_queries)
df .to_csv("annotation_bart.csv", index=False)
