import re
import ast
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from metrics import compute_score, recall, ndcg, mrr

# Formating functions
""" Clean the string by removing non-alphanumeric characters and converting to lowercase """
def clean_string(string:str) -> str:
    if not string:
        return ""
    return re.sub(r'\W+','', string).lower() 


# Loading functions
""" Format references list for trues """
def format_reference_trues(reference:dict) -> dict:
    return {
        "id_title": clean_string(reference["title"]),
        "id_s2": reference.get("id"),
        "year": reference["year"],
        "authors": [author["authorId"] for author in reference["authors"]],
        "citationCount": reference.get("citationCount")
    }

""" Format references list for preds """
def format_reference_preds(reference:dict) -> dict:
    return {
        "id_title": clean_string(reference["title"]),
        "id_s2": reference.get("id"),
        "year": reference["year"],
        "authors": reference.get("authors"),
        "citationCount": reference.get("citationCount")
    }

""" Select specific key from references list """
def select_keys_references(references:list, keys=["acl_s2"]) -> list:
    if len(keys) == 1:
        references = [reference[keys[0]] for reference in references]
    else :
        references = [{key: reference[key] for key in keys} for reference in references]
    return references

def load_trues(trues_path:str, keys=["id_s2"], acl_only=False) -> dict:
    reading_lists = pd.read_csv(trues_path)
    reading_lists['reading_list'] = reading_lists['reading_list'].apply(ast.literal_eval)

    trues = {}
    for key, references in zip(reading_lists["id"], reading_lists["reading_list"]):
        if acl_only:
            references = [format_reference_trues(reference) for reference in references if reference["in_acl"]]
        else:
            references = [format_reference_trues(reference) for reference in references]
        trues[key] = select_keys_references(references, keys=keys)
    return trues

""" Load predictions files based on field """ 
def load_predictions(preds_paths:list[str], annotator_ids = [1,2,3], keys=["id_s2"]) -> dict:
    preds_list = []
    for preds_path in preds_paths:
        preds = {}
        for annotator_id in annotator_ids:
            path_annots = Path(f'{preds_path}/preds_annot{annotator_id}.json')
            preds_annotator = {}
            for key, references in json.loads(path_annots.read_text()).items():
                references = [format_reference_preds(reference) for reference in references]
                preds_annotator[key] = select_keys_references(references, keys=keys)
            preds[f"A{annotator_id}"] = preds_annotator
        preds_list.append(preds)
    return preds_list


# Compute tables functions
""" Compute metrics bteween ground-thruth and predictions """ 
def process_scores(
        trues:dict, 
        preds_annotators:dict, 
        metrics:list=[recall, ndcg, mrr]
        ) -> dict:
    data = {}
    for k, preds in preds_annotators.items():
        data[k] = compute_score(trues, preds, metrics, k=20)
    return data

""" Select specific years valeus from predictions """ 
def select_year(preds_annotators:dict, year:int) -> dict:
    reading_lists = pd.read_csv("../reading_lists.csv")
    selected_ids = reading_lists[reading_lists["year"] == year]["id"].to_list()
    preds_year = {}
    for annotator_num, preds_annotator in preds_annotators.items():
        preds_year[annotator_num] = {id: preds_annotator[id] for id in selected_ids if id in preds_annotator}
    return preds_year

""" Compute metrics of predictionc for each year """  
def process_scores_years(
        trues:dict, 
        preds_annotators:dict, 
        metrics:list=[recall, ndcg, mrr]
    ) -> dict:
    years = [2020, 2021, 2022, 2023, 2024]
    data = {k: {} for k in years}
    for year in years:
        preds_annotators_year = select_year(preds_annotators, year)
        for k, preds_year in preds_annotators_year.items():
            data[year][k] = compute_score(trues, preds_year, metrics, k=20)
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
def compute_table_scores(
        model_scores:dict,
        metrics:list) -> pd.DataFrame:
    multi_columns = pd.MultiIndex.from_product([metrics,['A1', 'A2', 'A3', 'Mean']])
    df = pd.DataFrame(model_scores.values(), columns=multi_columns, index=model_scores.keys())
    return df

""" Compute DataFrame of models predictions per years """  
def compute_table_scores_years(
        models_scores:dict, 
        metrics:list
    ) -> pd.DataFrame:
    dfs = []
    for k, model_scores in models_scores.items():
        multi_columns = pd.MultiIndex.from_product([metrics,['A1', 'A2', 'A3', 'Mean']])
        multi_index = pd.MultiIndex.from_product([[k],['2020', '2021', '2022', '2023', '2024']])
        dfs.append(pd.DataFrame(model_scores, columns=multi_columns, index=multi_index))
    return pd.concat(dfs) 

""" Compute scores of models predictions """   
def score_models(
        trues:dict, 
        preds_list:list, 
        model_names:list,
        split_by_years:bool=False, 
        metrics:list=[recall, ndcg, mrr]
    ) -> pd.DataFrame:

    compute_table_scores_func = compute_table_scores_years if split_by_years else compute_table_scores
    convert_scores_to_list_func = convert_scores_to_list_years if split_by_years else convert_scores_to_list
    process_scores_func = process_scores_years if split_by_years else process_scores

    data = {}
    for model_name, preds_annotators in zip(model_names, preds_list):
        data[model_name] = convert_scores_to_list_func(process_scores_func(trues, preds_annotators, metrics=metrics))

    return compute_table_scores_func(data, [metric.__name__ for metric in metrics])


# Draw graphs functions
""" Draw a bar plot """    
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

""" Plot a distribution of years of publication for predicted articles"""
def draw_distribution_predictions(
    data:list,
    xlabel:str="", 
    ylabel:str="",
    figsize: tuple = (12, 6), 
    color:str ='#18dcff'
) -> None :
    unique, counts = np.unique(data, return_counts=True)
    
    #Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(unique, counts, color=color)
    for container in ax.containers:# Add bar labels
        ax.bar_label(container, padding=3, fontweight="bold", fontsize="x-small", color=color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(xlabel, fontsize="x-large", fontfamily="Helvetica Neue")
    ax.set_ylabel(ylabel, fontsize="x-large", fontfamily="Helvetica Neue")
    ax.tick_params(axis='both', which='major', labelsize="x-large", labelfontfamily="Helvetica Neue", direction="out")
    plt.show()