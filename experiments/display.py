import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

colormap = ['#18dcff', '#ffaf40', '#ff7979', "#a29bfe"]

# Set global font family and other font properties
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue'],
    'font.size': 16,
    'axes.titlesize': 30,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 18
})

def display_evolution_metric(table_years, keys, metric, colormap=colormap):
    data = defaultdict(list)
    years = [2020, 2021, 2022, 2023, 2024]
    for i, method in enumerate(keys):
        data["year"] = years
        for year in years:
            data[method].append(table_years.loc[(year,"Mean")][(method, metric)]/100)
    df = pd.DataFrame(data)
    ax = df.plot.bar(x="year", y=keys, ylabel=metric, color=colormap, figsize=(16,8), width=0.85)
    
    for i, container in enumerate(ax.containers):
        ax.bar_label(container, fmt="{:0.2f}", padding=3)
    
    ax.tick_params(axis='x', labelrotation=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("", fontsize=0)
    ax.set_ylabel("recall", fontsize=30)

def display_evolution_match_ids(mean_match, keys, colormap=colormap):
    data = defaultdict(list)
    years = [2020, 2021, 2022, 2023, 2024]
    for i, method in enumerate(keys):
        data["year"] = years
        for year in years:
            data[method].append(mean_match[str(year)][method])
    
    df = pd.DataFrame(data)
    ax = df.plot.bar(x="year", y=keys, ylabel="matches", color=colormap, figsize=(16,8), width=0.85)
    
    for i, container in enumerate(ax.containers):
        ax.bar_label(container, fmt="{:0.2f}", padding=3)
    
    ax.tick_params(axis='x', labelrotation=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("", fontsize=0)
    ax.set_ylabel("matches", fontsize=30)