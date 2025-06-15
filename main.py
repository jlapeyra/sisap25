import argparse
from dataclasses import dataclass
import heapq
from typing import Callable, Literal, Type
from scipy.spatial import KDTree, distance_matrix
import numpy as np
import h5py
from pca import PCA, PCA_sklearn, PCA_faiss
from timer import timer, timed
from utils import format_num
import faiss
from collections import Counter
import os
from pathlib import Path
from datasets import DATASETS, prepare, get_fn
from metadata import get_k
import json
import threading
from eval import load_solution, get_recall
from logger import log, time_since_start
import math
import logger

NUM_THREADS = 6
RAM_LIMIT = 16 * 1024**3  # 16 GB
BATCH_SIZE = 20_000

@dataclass
class Params:
    d_pca:int
    k_search:int
    no_self_loops=False
    def to_dict(self):
        return {
            'd_pca': self.d_pca,
            'k_search': self.k_search,
            'no_self_loops': self.no_self_loops,
        }
        
def defaultParams(task) -> Params:
    match task:
        case 'task1': return params_task1
        case 'task2': return params_task2


def store_results(dst, dataset, task, I:np.ndarray, D:np.ndarray, params:'Params'):
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, 'w')

    assert I.shape == D.shape, "Shapes of I and D must match"
    assert I.ndim == 2, "I must be a 2D array"
    assert D.ndim == 2, "D must be a 2D array"
    size = I.shape[0]

    f.attrs['dataset'] = dataset
    f.attrs['task'] = task
    f.attrs['time'] = time_since_start()
    f.attrs['params'] = json.dumps(params.to_dict())
    f.attrs['num_queries'] = size

    f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
    f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
    f.close()

@timed
def load_data(dataset, task):
    global data_file

    fn, _ = get_fn(dataset, task)
    data_file = h5py.File(fn)
    data = DATASETS[dataset][task]['data'](data_file)
    if task == 'task1':
        queries = DATASETS[dataset][task]['queries'](data_file)
    else:
        queries = data
    logger.log(f'Loaded {len(data)} data points, {len(queries)} queries')
    return data, queries




def index_search(index:faiss.Index, queries:np.ndarray, k:int, pca:PCA=None):
    if pca is not None:
        queries = pca.transform(queries)
    # Otherwise, it is assumed that queries are already PCA transformed
    _, found = index.search(queries, k=k)
    return found

#def multi_threaded_search(index:faiss.Index, queries:np.ndarray, k:int, pca:PCA=None, n_threads:int=NUM_THREADS):

def search(
    I:np.ndarray,
    D:np.ndarray,
    previous_candidates:bool,
    index:faiss.Index, 
    data:np.ndarray,
    queries:np.ndarray,
    pca_queries:np.ndarray|None,
    k:int,
    k_search:int,
    no_self_loops:bool,
    query_slice:slice = slice(0, None),
    data_slice:slice = slice(0, None),
    pca:PCA|None=None,
):
    if pca_queries is not None:
        search_queries = pca_queries
    else:
        search_queries = queries
        #assert pca is not None


    N = len(search_queries)
    n_threads = min(NUM_THREADS, N)  # Limit the number of threads to the number of queries

    threads = []


    #@timed
    def search_part(thread_id):
        start = thread_id * (N // n_threads)
        end = (thread_id + 1) * (N // n_threads) if thread_id < n_threads - 1 else N
        with timer(f'index_search[{thread_id}]'): 
            found = index_search(
                index, 
                search_queries[start:end], 
                k_search, 
                pca
            )
        with timer(f'choose[{thread_id}]'): 
            choose(
                I, D, previous_candidates,
                data, 
                queries[start:end], 
                found, 
                k,
                no_self_loops,
                query_slice=slice(query_slice.start + start, query_slice.start + end),
                data_slice=data_slice,
            )

    for i in range(n_threads):
        thread = threading.Thread(target=search_part, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


def get_distance(v1:np.ndarray, v2:np.ndarray):
    return -np.dot(v1, v2)

#is_itself:Callable[[int,int], bool],
def choose(
        I:np.ndarray, D:np.ndarray, previous_candidates:bool, 
        data:np.ndarray, queries:np.ndarray, found:np.ndarray, k:int, no_self_loops:bool,
        query_slice:slice = slice(0, None), data_slice:slice = slice(0, None), 
    ):
    """
    Choose the k nearest neighbors from the found indices, ensuring that the query itself is not included.
    """
    for i_query, (query, nn_found) in enumerate(zip(queries, found), start=query_slice.start):
        candidates = []
        if previous_candidates:
            candidates = list(zip(D[i_query, :], I[i_query, :]))
        #dist = distance_matrix([query], [data[nn] for nn in nn_found])[0]
        candidates.extend([
            (get_distance(data[i_nn], query), i_nn + data_slice.start + 1) #+1 because groundtruth is 1-indexed 
            for i_nn in nn_found
            if not (no_self_loops and i_query == i_nn + data_slice.start)
        ])
        closest = heapq.nsmallest(k, candidates, key=lambda x: x[0])
        I[i_query, :] = [i for d,i in closest]
        D[i_query, :] = [d for d,i in closest]


def pca_and_index(
    data:np.ndarray,
    d_pca:int|None,
    pca_class:Type[PCA] = PCA_sklearn,
):
    _, d = data.shape

    if d_pca is not None:
        with timer('PCA'):
            pca = pca_class(d, d_pca)
            data = pca.fit_transform(data)
    else:
        pca = None

    with timer('index'):
        index = faiss.IndexFlatIP(data.shape[1])  
        index.add(data)

    return pca, data, index


@timed
def task1(
    dataset:str,
    data:np.ndarray,
    queries:np.ndarray,
    params:'Params', 
    dst:str,
):
    # The assumed dataset is 'plumbed23', with 36 GB
    # The RAM has 16 GB, so we cannot load the whole dataset into memory


    task = 'task1'
    k = get_k(task)
    logger.log(len(data), 'data points,', len(queries), 'queries')
    N = len(data)

    data_size = N * data.shape[1] * 4
    global BATCH_SIZE
    if BATCH_SIZE:
        num_batches = math.ceil(data.shape[0]/BATCH_SIZE)
    else:
        num_batches = math.ceil(data_size/(RAM_LIMIT/4))

    N_Q = len(queries)
    I, D = np.zeros((N_Q, k), dtype=np.int32), np.zeros((N_Q, k), dtype=np.float32)
    batch_size = N // num_batches
    logger.start_loop_time()
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size if i < num_batches - 1 else N
        logger.log(f'Processing batch {i+1}/{num_batches} ({start}:{end})')

        sliced_data = data[start:end]

        pca, pca_data, index = pca_and_index(
            sliced_data, 
            d_pca=params.d_pca
        )

        search(
            I, D, (i>0),
            index, 
            sliced_data,
            queries=queries,
            pca_queries=None,
            k=k,
            k_search=params.k_search,
            no_self_loops=False,
            data_slice=slice(start, end),
            pca=pca,
        )

        store_results(
            dst=dst,
            dataset=dataset,
            task=task,
            I=I, 
            D=D,
            params=params
        )
        get_recall(task, dataset, I, data_coverage=end/len(data), data_size=len(data))
        logger.log_expected_time(i, num_batches)
    logger.stop_loop_time()
    return I, D

@timed
def task2(
    dataset:str,
    data:np.ndarray, 
    params:'Params',
    dst:str,
):
    # The assumed dataset is 'gooaq', with 4.7 GB
    # The RAM has 16 GB, so we can load the whole dataset into memory
    task = 'task2'
    k = get_k(task)
    if not params.no_self_loops:
        k += 1 # if we allow self loops, we must consider one element more
    N = len(data)
    pca, pca_data, index = pca_and_index(
        data, 
        d_pca=params.d_pca
    )
    I, D = np.zeros((N, k), dtype=np.int32), np.zeros((N, k), dtype=np.float32)

    global BATCH_SIZE
    if BATCH_SIZE:
        num_batches = math.ceil(N/BATCH_SIZE)
    else:
        num_batches = math.ceil(N/150_000)
    batch_size = N // num_batches
    logger.start_loop_time()
    for i in range(num_batches):
        start = i * batch_size
        end = (i + 1) * batch_size if i < num_batches - 1 else N
        logger.log(f'Processing batch {i+1}/{num_batches} ({start}:{end})')
        query_slice = slice(start, end)
        search(
            I, D, False,
            index, 
            data,
            queries=data[query_slice],
            pca_queries=pca_data[query_slice],  # queries are the same as data in task2
            k=k,
            k_search=params.k_search,
            no_self_loops=params.no_self_loops,
            query_slice=query_slice,
        )

        store_results(
            dst,
            dataset=dataset,
            task=task,
            I=I[:end], 
            D=D[:end],
            params=params
        )
        get_recall(task, dataset, I[:end], no_self_loops=params.no_self_loops)
        logger.log_expected_time(i, num_batches)
    logger.start_loop_time()
    return I, D




@timed
def main(
    task:Literal['task1', 'task2'], 
    dataset:str,
    params:'Params'=None,
):

    logger.log(f'Running {task} on {dataset}')

    if params is None:
        params = defaultParams(task)

    logger.log(params.to_dict())

    prepare(dataset, task)
    data, queries = load_data(dataset, task)

    os.makedirs('results', exist_ok=True)
    dst = f'results/{dataset}_{task}_d={params.d_pca}_k={params.k_search}.h5'

    if task == 'task1':
        task1(dataset, data, queries, params, dst)
    elif task == 'task2':
        task2(dataset, data, params, dst)


params_task2 = Params(
    d_pca = 140,
    k_search = 200
)
params_task1 = Params(
    d_pca = None,
    k_search = 200,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=['task1', 'task2'],
        default='task1'
    )

    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        default='pubmed23'
    )

    args = parser.parse_args()
    main(args.task, args.dataset)









#logger.log(d)
#logger.log(i)



