# Vector Search Evaluation for SISAP 2025

This project implements and evaluates nearest neighbor search algorithms for high-dimensional datasets, following the format and requirements of the [SISAP 2025 Nearest Neighbor Search Challenge](https://sisap-challenges.github.io/2025/).

It supports two tasks:

* **Task 1**: Approximate k-NN search with disjoint training and query sets.
* **Task 2**: All-k-NN search within the same dataset (queries = data).

---

## 📁 Project Structure

* `main.py`: Entry point for both tasks.
* `datasets.py`: Handles dataset loading, linking/downloading, and structure validation.
* `eval.py`: Computes recall based on ground-truth and evaluates search results.
* `metadata.py`: Defines dataset metadata and parsing logic.
* `timer.py`: Utility for measuring and logging execution time.
* `logger.py` and `utils.py`: Logging and formatting utilities.
* `pca.py`: Implements PCA variants (e.g., FAISS, sklearn).

---

## ✅ Requirements

Install the required libraries:

```bash
pip install -r requirements.txt
```


## 🧪 Running Experiments

Prepare data first:

```bash
python datasets.py
```

Then, to run a search task:

```bash
python main.py --task task1 --dataset pubmed23
```

```bash
python main.py --task task2 --dataset gooaq
```

Options:

* `--task`: `task1` or `task2`
* `--dataset`: A dataset present in the data/ directory (e.g., `gooaq`, `ccnews`, `pubmed23`)

Results will be stored in the `results/` directory with appropriate metadata.

---

## 📊 Evaluation

To evaluate recall:

```bash
python eval.py
```

It reads the results file and computes recall by comparing against the ground truth stored in `data/[dataset]/task[1|2]/gt/`.

---

## 📦 Dataset Format

This project expects datasets to follow the SISAP structure:

```
data/
  └── [dataset]/
       ├── task1/
       │    ├── [dataset].h5
       │    └── gt/
       │         └── gt_[dataset].h5
       └── task2/
            ├── [dataset].h5
            └── gt/
                 └── gt_[dataset].h5
```

If your data is not organized this way, you can specify custom paths in `metadata.py`.

---

## 🛠 Parameters

You can tweak search parameters like PCA dimension (`d_pca`) and search depth (`k_search`) in `main.py` by editing `params_task1` and `params_task2`.

---

## 🧠 Notes

* PCA-based dimensionality reduction is optional and supports both `sklearn` and `faiss` implementations.
* Multithreading is used to parallelize query processing.
* Results include recall metrics and runtime stats for benchmarking.

