import numpy as np
import math


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    recalls = []
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            recalls.append(len(act_set & pred_set) / float(len(act_set)))
            true_users += 1
    # print(recalls[: 20])
    return sum_recall / true_users


def ndcg_recall_at_k(actual, predicted, topk):
    r = []
    for i in predicted:
        if i in actual:
            r.append(1)
        else:
            r.append(0)
    r = np.asfarray(r)[:topk]
    return np.sum(r) / len(actual)


def batch_ndcg_recall_at_k(actual, predicted, k):
    result = 0.0
    assert len(actual) == len(predicted)
    num_users = len(actual)

    for user_id in range(len(actual)):
        result += ndcg_recall_at_k(actual[user_id], predicted[user_id], k)
    return result / num_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


# Calculates the normalized discounted cumulative gain at k
# https://github.com/makgyver/pyros/blob/master/pyros/core/evaluation.py
def ndcg_k(actual, predicted, topk):
    # k = min(k, len(predicted))
    idcg = idcg_k(actual, topk)
    res = 0
    for user_id in range(len(actual)):
        dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(actual, k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(min(k, len(actual)))])
    if not res:
        return 1.0
    else:
        return res


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(actual, predicted, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    r = []
    for i in predicted:
        if i in actual:
            r.append(1)
        else:
            r.append(0)
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def batch_ndcg_at_k(actual, predicted, k):
    result = 0.0
    assert len(actual) == len(predicted)
    num_users = len(actual)

    for user_id in range(len(actual)):
        result += ndcg_at_k(actual[user_id], predicted[user_id], k)
    return result / num_users

