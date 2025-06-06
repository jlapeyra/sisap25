'''
INSTRUCTIONS TO LOAD YOUR DATA FILES:

The main program will read data from:
    data/[dataset]/task1/[dataset].h5
    data/[dataset]/task1/gt/gt_[dataset].h5
    data/[dataset]/task2/[dataset].h5
    data/[dataset]/task2/gt/gt_[dataset].h5
Which is the file structure specified on https://sisap-challenges.github.io/2025/index.html#examples

If you have the said structure, you do not need to change this file, unless the internal structure
of the h5 files is different from the one specified in the `METADATA` variable below. If this is the case, 
you should change the `METADATA` variable below to match your data structure, or specify a different 
`metadata` in `Metadata` in your entry of `__DATASETS`.


If your data is not there, you should add to the variable `__DATASETS` below an entry with:
 - Local path or URL of data file, with train data and queries
 - Local path or URL of allknn file, with the solution of task2 (only required to evaluate task2)

If you provide a URL the files will be downloaded to the mentioned directories.
If you provide a local path, the files will be linked in order to prevent redundant storage in your computer.

Data files are assumed to be in the format of the examples provided in https://huggingface.co/datasets/sadit/SISAP2025
In particular,
    - Task 1 will train with x['train'] and query from x['otest']['queries'],  where x = data/[dataset]/task1/[dataset].h5
    - Task 1 will be evaluated based on gold standard from x['otest']['knns'], where x = data/[dataset]/task1/gt/gt_[dataset].h5
    - Task 2 will train with x['train'] and query from x['train'],             where x = data/[dataset]/task2/[dataset].h5
    - Task 2 will be evaluated based on gold standard from x['knns'],          where x = data/[dataset]/task2/gt/gt_[dataset].h5
If your data file is in another format or you wish to use some other queries in task 1 (e.g. 'itest' instead of 'otest'),
make the according changes in the `METADATA` variable or specify a different `metadata` in `Metadata` in your entry of `__DATASETS`

'''

from copy import deepcopy
from collections import defaultdict
import os
import glob

METADATA = {
    'task1': {
        'data':       lambda x: x['train'],
        'queries':    lambda x: x['otest']['queries'],
        'solution':   lambda x: x['otest']['knns'],
        #'queries':  lambda x: x['itest']['queries'],
        #'solution': lambda x: x['itest'].get('knns'),
    },
    'task2': {
        'data':     lambda x: x['train'],
        'solution': lambda x: x['knns'],
    }
}


def get_k(task):
    if task == 'task1':
        return 30
    elif task == 'task2':
        return 15
    else:
        raise ValueError(f"Unknown task: {task}. Valid tasks are 'task1' and 'task2'.")

class Metadata(dict):
    def __init__(self, src=None, allknn_src=None, metadata=METADATA):
        self.update(deepcopy(metadata))
        self['task1']['src'] = src
        self['task1']['gt_src'] = src
        self['task2']['src'] = src
        self['task2']['gt_src'] = allknn_src





#'https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/benchmark-dev-ccnews-fp16.h5?download=true',
#'https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/allknn-benchmark-dev-ccnews.h5?download=true',

__DATASETS = {

}

def create_metadata_entries(data_dir='data'):
    entries = {}
    if not os.path.isdir(data_dir):
        return entries
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            entries[folder] = Metadata()
    return entries

DATASETS = {}
DATASETS.update(create_metadata_entries())
# DATASETS.update({
#     'ccnews' : Metadata(
#         src='data/ccnews.h5', 
#         allknn_src='data/allknn-ccnews.h5'
#     ),
#     'gooaq' : Metadata(
#         src='data/gooaq.h5',
#         allknn_src='data/allknn-gooaq.h5'
#     ),
#     'pubmed23' : Metadata(
#         src='data/pubmed23.h5',
#     ),
# })






