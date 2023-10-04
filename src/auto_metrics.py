from transformers import EvalPrediction
import numpy as np
import os

def ndcg(y_true, y_pred, k=None, powered=False):
    def dcg(scores, k=None, powered=False):
        if k is None:
            k = scores.shape[0]
        if not powered:
            ret = scores[0]
            for i in range(1, k):
                ret += scores[i] / np.log2(i + 1)
            return ret
        else:
            ret = 0
            for i in range(k):
                ret += (2 ** scores[i] - 1) / np.log2(i + 2)
            return ret

    ideal_sorted_scores = np.sort(y_true)[::-1]
    ideal_dcg_score = dcg(ideal_sorted_scores, k=k, powered=powered)

    pred_sorted_ind = np.argsort(y_pred)[::-1]
    pred_sorted_scores = y_true[pred_sorted_ind]
    dcg_score = dcg(pred_sorted_scores, k=k, powered=powered)

    return dcg_score / ideal_dcg_score

def ndcg1(y_true, y_pred, k=None):
    return ndcg(y_true, y_pred, k=k, powered=False)

def ndcg2(y_true, y_pred, k=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tmp_dict = {}
    for i in range(1, k+1):
        tmp_dict[str(i)] = ndcg(y_true, y_pred, k=i, powered=True)
    return tmp_dict


def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        # if method == 0:
        #     return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        # elif method == 1:
        #     return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        # else:
        #     raise ValueError('method must be 0 or 1.')
        return np.sum((np.power(2, r)-1) / np.log2(np.arange(1, r.size + 1)+1))
    return 0.


def ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def ndcg_dict(pred, ans, max_k=None):
    if max_k is None:
        max_k = len(pred)
    if pred is not np:
        x = np.array(pred)
    if ans is not np:
        y = np.array(ans)


    sorted_index = np.argsort(-x)
    rs = y[sorted_index].tolist()

    ndcgs = {}
    for i in range(1, max_k + 1):
        ndcgs[i] = 0.0

    for i in range(1, max_k + 1):
        ndcgs[i] = ndcg_at_k(rs, i)
    return ndcgs

def custom_compute_metrics(res: EvalPrediction) -> Dict:
    # res.predictions, res.label_idsはnumpyのarray
    pred = res.predictions
    target = res.label_ids
