import math
import numpy as np
from collections import defaultdict

def recall(trues, preds):
    trues = list(dict.fromkeys(trues))
    preds = list(dict.fromkeys(preds))
    matches = set(trues) & set(preds)
    return len(matches)/len(trues)

def ndcg(trues, preds):
    trues = list(dict.fromkeys(trues))
    preds = list(dict.fromkeys(preds))
    matches = set(trues) & set(preds)
    dcg = np.sum([1/math.log2(preds.index(match)+2) for match in matches])
    dcg_i = np.sum([1/math.log2(i+2) for i in range(0, len(trues))])
    return dcg/dcg_i

def mrr(trues, preds):
    trues = list(dict.fromkeys(trues))
    preds = list(dict.fromkeys(preds))
    matches = set(trues) & set(preds)
    return 1/([i for i, pred in enumerate(preds) if pred in matches][0]+1) if len(matches) else 0

def score(trues, preds, metrics, k=20):
    scores = defaultdict(list)
    for key, pred in preds.items():
        true = trues[key]
        if len(true):
            for metric in metrics:
                scores[metric.__name__].append(metric(true, pred[0:k]) * 100)
    return {k: np.mean(v) for k, v in scores.items()}