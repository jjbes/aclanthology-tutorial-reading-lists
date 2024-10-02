import re
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

from metrics import compute_score, recall, ndcg, mrr

""" Clean the string by removing non-alphanumeric characters and converting to lowercase """
def clean_string(string:str) -> str:
    if not string:
        return ""
    return re.sub(r'\W+','', string).lower() 

""" Load predictions files based on field """ 
def load_preds(preds_path:str, use_title_instead_of_id:bool=False) -> dict:
    preds = defaultdict(lambda: defaultdict(dict))
    for annotator_i in [1,2,3]:
        path_annots = Path(f'{preds_path}/preds_annot{annotator_i}.json')
        if use_title_instead_of_id:
            preds[f"A{annotator_i}"] = { id_:[clean_string(ref["title"]) for ref in references] for id_, references in json.loads(path_annots.read_text()).items()}
        else:
            preds[f"A{annotator_i}"] = { id_:[ref["id"] for ref in references] for id_, references in json.loads(path_annots.read_text()).items()}

    return json.loads(json.dumps(preds))

""" Compute metrics bteween ground-thruth and predictions """ 
def process_scores(trues:dict, preds_dict:dict) -> dict:
    data = {}
    for k, preds in preds_dict.items():
        data[k] = compute_score(trues, preds, [recall, ndcg, mrr], k=20)
    return data

""" Select specific years valeus from predictions """ 
def select_year(preds_dict:dict, year:int) -> dict:
    reading_lists = pd.read_csv("../reading_lists.csv")
    selected_ids = reading_lists[reading_lists["year"] == year]["id"].to_list()
    preds_year = {}
    for annotator_num, preds_annotator in preds_dict.items():
        preds_year[annotator_num] = {id: preds_annotator[id] for id in selected_ids if id in preds_annotator}
    return preds_year

""" Compute metrics of predictionc for each year """  
def process_scores_years(trues:dict, preds_dict:dict) -> dict:
    years = [2020, 2021, 2022, 2023, 2024]
    data = {k: {} for k in years}
    for year in years:
        preds_dict_year = select_year(preds_dict, year)
        for k, preds_year in preds_dict_year.items():
            data[year][k] = compute_score(trues, preds_year, [recall, ndcg, mrr], k=20)
    return data

""" Convert a dict of scores to a flat list """  
def convert_scores_to_list(data:dict) -> list:
    df = pd.DataFrame(data)
    mean_values = df.mean(axis=1)
    df['mean'] = mean_values
    return df.values.flatten().tolist()
    
""" Compute a dict of scores to a flat list for each year """  
def convert_scores_to_list_years(data:dict) -> dict:
    years = [2020, 2021, 2022, 2023, 2024]
    score_list_years = []
    for year in years:
        df = pd.DataFrame(data[year])
        mean_values = df.mean(axis=1)
        df['mean'] = mean_values
        score_list_years.append(df.values.flatten().tolist())
    return score_list_years
    
""" Compute DataFrame of models predictions """  
def compute_table_scores(model_scores:dict) -> pd.DataFrame:
    multi_columns = pd.MultiIndex.from_product([['recall', 'ndcg', 'mrr'],['A1', 'A2', 'A3', 'Mean']])
    df = pd.DataFrame(model_scores.values(), columns=multi_columns, index=model_scores.keys())
    return df

""" Compute DataFrame of models predictions per years """  
def compute_table_scores_years(models_scores:dict) -> pd.DataFrame:
    dfs = []
    for k, model_scores in models_scores.items():
        multi_columns = pd.MultiIndex.from_product([['recall', 'ndcg', 'mrr'],['A1', 'A2', 'A3', 'Mean']])
        multi_index = pd.MultiIndex.from_product([[k],['2020', '2021', '2022', '2023', '2024']])
        dfs.append(pd.DataFrame(model_scores, columns=multi_columns, index=multi_index))
    return pd.concat(dfs) 

""" Compute scores of models predictions """   
def score_models(
        trues:dict, 
        models:dict, 
        paths:str,
        split_by_years:bool=False, 
        use_title_instead_of_id:bool=False
    ) -> pd.DataFrame:

    compute_table_scores_func = compute_table_scores_years if split_by_years else compute_table_scores
    convert_scores_to_list_func = convert_scores_to_list_years if split_by_years else convert_scores_to_list
    process_scores_func = process_scores_years if split_by_years else process_scores
    return compute_table_scores_func({model:convert_scores_to_list_func(process_scores_func(trues, load_preds(path,use_title_instead_of_id=use_title_instead_of_id))) for model, path in zip(models, paths)})

""" Draw an histogram plot """    
def draw_bar_plot(
        df:pd.DataFrame, 
        xlabel:str="", 
        ylabel:str="", 
        figsize: tuple = (12, 6), 
        colormap:list =['#18dcff', '#ffaf40', '#ff7979', "#a29bfe", "#2bebb6"]
    ) -> None:

    ax = df.plot.bar(x="year", color=colormap, figsize=figsize, width=0.85)
    for i, container in enumerate(ax.containers):
        ax.bar_label(
            container, 
            fmt="{:.2f}", 
            padding=3, 
            fontweight ="bold", 
            color=colormap[i],
            fontsize=round(40/len(ax.containers))
        )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(xlabel, fontsize="xx-large")
    ax.set_ylabel(ylabel, fontsize="xx-large", fontfamily="Helvetica Neue")
    ax.tick_params(axis='x', labelrotation=0)
    ax.tick_params(axis='both', which='major', labelsize="xx-large", labelfontfamily="Helvetica Neue", direction="out")
    plt.show()