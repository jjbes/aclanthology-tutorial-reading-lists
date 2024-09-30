import numpy as np
import typing_extensions as typing 
from collections import defaultdict

def recall(trues:list, preds:list) -> float:
    """ Measure recall between a ground-thruth and predictions """ 
    matches = set(trues) & set(preds)
    return len(matches)/len(trues)

def ndcg(trues:list, preds:list) -> float:
    """ Measure NDCG between a ground-thruth and predictions """ 
    #Binary relevance -> could be graded for partial matching of articles (what would be a partial match?)
    relevance_scores = [1 if item in trues else 0 for item in preds]
    dcg  = np.sum([(pow(2,score)-1)/np.log2(i+1) for i, score in enumerate(relevance_scores, start=1)])
    idcg = np.sum([(pow(2,1)-1)/np.log2(i+1) for i, _ in enumerate(trues, start=1)])
    # Avoid division by zero
    return (dcg / idcg) if idcg > 0 else 0

def mrr(trues:list, preds:list) -> float:
    """ Measure Mean Reciprocal Rank between a ground-thruth and predictions """ 
    for i, pred in enumerate(preds, start=1):
        if pred in trues:
            return 1 / i
    return 0

def compute_score(trues:list, preds:list, metrics:list[typing.Callable], k:int=20) -> typing.Dict:
    """ Compute multiple metrics @k """ 
    scores = defaultdict(list)
    for key, pred in preds.items():
        true = trues[key]
        if len(true):
            for metric in metrics:
                scores[metric.__name__].append(metric(true, pred[0:k]) * 100)
    return {k: np.mean(v) for k, v in scores.items()}