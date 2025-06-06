import h5py
import os 
from urllib.request import urlretrieve
from pathlib import Path
import shutil
from metadata import DATASETS
import numpy as np
from logger import log

class DataNotFoundError(Exception):
    pass

def data_file_exists(src):
    if os.path.exists(src):
        return True
    elif os.path.exists(src+'.link'):
        with open(src+'.link') as f:
            return os.path.exists(f.read())
    else:
        return False


def download_or_link(src:str, dst:str):
    if not data_file_exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        if src.startswith('http'): #download
            log('downloading %s -> %s...' % (src, dst))
            urlretrieve(src, dst)
        else: #link
            with open(dst+'.link', 'w') as f:
                f.write(os.path.abspath(src))
            # avoid unnecessary duplication of files
            #log('copying %s -> %s...' % (src, dst))
            #shutil.copy(src, dst)

def get_canonic_fn(dataset, task):
    return os.path.join("data", dataset, task, f"{dataset}.h5"), os.path.join('data', dataset, task, 'gt', f'gt_{dataset}.h5')
    
def get_fn(dataset, task):
    path = list(get_canonic_fn(dataset, task))
    for i in range(2):
        if not os.path.exists(path[i]):
            with open(path[i]+'.link') as f:
                path[i] = f.read().strip()
    return tuple(path)


def prepare(dataset, task, ignore_gt=False):
    fn, gt_fn = get_canonic_fn(dataset, task)
    src = DATASETS[dataset][task]['src'] #url or path
    gt_src = DATASETS[dataset][task]['gt_src'] #url or path

    if src is None and not data_file_exists(fn):
        raise DataNotFoundError('\n'
            f'\tCould not found main data for {task} and dataset {dataset}.\n'
            f'\tSpecify their origin (path or url) at `metadata.py`, following the instructions there.\n'
            f'\tAlternatively, copy the file to {fn}.'
        )
    if not ignore_gt and gt_src is None and not data_file_exists(fn):
        raise DataNotFoundError(
            f'\tCould not found gt data for {task} and dataset {dataset}.\n'
            f'\tSpecify their origin (path or url) at `metadata.py`, following the instructions there.\n'
            f'\tAlternatively, copy the file to {fn}.'
        )

    download_or_link(src, fn)
    if not ignore_gt:
        download_or_link(gt_src, gt_fn)

def get_query_count(dataset, task):
    fn, _ = get_fn(dataset, task) 
    f = h5py.File(fn)
    qn = len(DATASETS[dataset][task]['queries'](f))
    f.close()
    return qn


if __name__ == '__main__':
    prepare('gooaq', 'task1')
    prepare('gooaq', 'task2')
    # prepare('ccnews', 'task1')
    # prepare('ccnews', 'task2')
