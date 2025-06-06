import argparse
import numpy as np
import h5py
from datasets import DATASETS, get_fn
from utils import format_num
import json
from metadata import get_k
from logger import log, Colors

def load_solution(dataset, task):
    global sol_file
    _, fn_sol = get_fn(dataset, task)
    sol_file = h5py.File(fn_sol)
    try:
        solution = np.array(DATASETS[dataset][task]['solution'](sol_file))
    except KeyError:
        raise KeyError(f'No solution found for {dataset} and {task}. Check your metadata.py file.')
    return solution

def get_recall(task, dataset, results, data_coverage:float=1.0, data_size:int=None, no_self_loops=False):
    solution = load_solution(dataset, task)
    
    total_queries = len(solution)
    analized_queries = len(results)
    rate = analized_queries / total_queries
    
    self_loops = task == 'task2' and not no_self_loops

    k = results.shape[1] - self_loops

    if k != get_k(task):
        log(Colors.yellow(f'WARNING: Expected k={get_k(task)} for task {task}, but got k={k} in results file {fn}.'))

    if task == 'task2':
        solution = solution[:, 1:]
    if self_loops:
        results = results[:, 1:]
    solution = solution[:, :k]

    recall = np.mean([len(set(r) & set(s)) for r, s in zip(results, solution)]) / k


    log()
    if analized_queries != total_queries:
        log(format_num(analized_queries), 'of', format_num(total_queries), 'queries',  f'({100*rate:.1f}%)',)
    log('mean hits =',recall*k, '| k =', k, '| recall =', recall)
    #assert 0 < data_coverage <= 1
    if data_coverage < 1:
        if data_size is None:
            log(f'Data coverage: {data_coverage*100:.1f}%')
        else:
            log(format_num(data_coverage*data_size), 'of', format_num(data_size), 'data points',  f'({data_coverage*100:.1f}%)',)
        log(f'normalized recall = {recall/data_coverage}')
    log()
    
    return recall

def load_results(fn):
    log(f'Loading results from {fn}...')
    f = h5py.File(fn, 'r')
    I = f['knns']
    #D = f['dists']
    params:dict = json.loads(f.attrs.get('params') or '{}')
    dataset = f.attrs['dataset']
    task = f.attrs['task']
    num_queries = I.shape[0]

    log(f'Loaded {num_queries} queries for {dataset} and {task}')
    
    return I, dataset, task, params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
    )
    args = parser.parse_args()
    fn = args.results
    results, dataset, task, params = load_results(fn)
    log(params)
    get_recall(task, dataset, results, no_self_loops=params.get('no_self_loops', False))

