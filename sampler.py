import random
import numpy as np
from multiprocessing import Process, Queue
import gc


def pad_adj(adj, max_degree=128, n_neighbour=800):
    '''
    Get neighbor certain number of neighbors from neighbor dictionary
    :param adj: Dictionary of neighbors
    :param max_degree: How many neighbors to keep at most
    :param n_neighbour: maximum number of neighbors to choose for random selecting neighbors
    :return: array of neighbor arrays
    :param seed: reproduce random behaviour
    '''
    adj = adj.copy()
    if n_neighbour == 0:
        n_neighbour = 1000
    for nodeid in adj.keys():
        neighbors = list(adj[nodeid])

        if len(neighbors) == 0:
            neighbors = np.random.choice(n_neighbour, max_degree, replace=True)
        else:
            if len(neighbors) > max_degree:
                neighbors = np.random.choice(neighbors, max_degree, replace=False)
            elif len(neighbors) < max_degree:
                neighbors = np.random.choice(neighbors, max_degree, replace=True)
        adj[nodeid] = neighbors
    return np.stack([*adj.values()], axis=0)


def fast_shuffle(series, num):
    length = series.shape[1]
    for i in range(num):
        yield series[:, np.random.permutation(length)]


def fast_shuffle_2(series):
    random = series.T.copy()
    np.random.shuffle(random)
    return random.T


def sample_function(user_item_edge, n_item, batch_size, n_negative, result_queue, check_negative=True):
    """
    :param user_item_matrix: the user-item matrix for positive user-item pairs
    :param batch_size: number of samples to return
    :param n_negative: number of negative samples per user-positive-item pair
    :param result_queue: the output queue
    :return: None
    """
    user_item_pairs = []
    for i in range(len(user_item_edge)):
        user_item_pairs += [[i, j] for j in user_item_edge[i]]
    user_item_pairs = np.array(user_item_pairs)
    user_to_positive_set = {u: set(row) for u, row in enumerate(user_item_edge)}

    while True:
        np.random.shuffle(user_item_pairs)
        for i in range(int(len(user_item_pairs) / batch_size)):

            user_positive_items_pairs = user_item_pairs[i * batch_size: (i + 1) * batch_size, :]

            # sample negative samples
            negative_samples = np.random.randint(
                0,
                n_item,
                size=(batch_size, n_negative))

            if check_negative:
                for user_positive, negatives, i in zip(user_positive_items_pairs,
                                                       negative_samples,
                                                       range(len(negative_samples))):
                    user = user_positive[0]
                    for j, neg in enumerate(negatives):
                        while neg in user_to_positive_set[user]:
                            negative_samples[i, j] = neg = np.random.randint(0, n_item)
            result_queue.put((user_positive_items_pairs, negative_samples))


class WarpSampler(object):
    """
    A generator that, in parallel, generates tuples: user-positive-item pairs, negative-items
    of the shapes (Batch Size, 2) and (Batch Size, N_Negative)
    """

    def __init__(self, user_item_matrix, n_item, batch_size=10000, n_negative=10, n_workers=1, check_negative=True,
                 seed=0):
        gc.collect()
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(user_item_matrix,
                                                      n_item,
                                                      batch_size,
                                                      n_negative,
                                                      self.result_queue,
                                                      check_negative)))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get(timeout=60)

    def close(self):
        for p in self.processors:  # type: Process
            p.terminate()
            p.join()
        self.result_queue.close()
        self.result_queue.join_thread()
