# DoReMi🎶: Domain Reweighting with Minimax Optimization
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2305.10429-00ff00.svg)](https://arxiv.org/abs/2305.10429)

PyTorch implementation of DoReMi, an algorithm for optimizing data mixtures for language modeling datasets. Check out the [paper](https://arxiv.org/abs/2305.10429) for more details. This repo is currently in active development!

![High-level overview of DoReMi.](doremi.gif)

Note that there may be a few differences between this repo and the paper, which was developed at Google, namely:
- PyTorch vs JAX
- Subtle differences in model architecture
- Tokenizers used (256k vocab size used in paper, while standard GPT2 tokenizer is 50k vocab size). This can siginificantly affect the data mixtures as calculated by token count.
You should run DoReMi within your own specific training setup for the best results.

## Getting started

To get started, please clone the repo, and install it:
```
git clone git@github.com:/sangmichaelxie/doremi.git
pip install -e doremi
cd doremi && bash scripts/setup_flash.sh
```

All code should be run from the outermost `doremi` directory.
Before you start, write paths to your cache directories, data directories, etc in a `constants.sh` file in the outer directory of this repo. You can also place any conda or virtualenv activation commands here. Here's an example of the contents of a `constants.sh` file:
```
#!/bin/bash
CACHE=/path/to/cache
DOREMI_DIR=/path/to/this/repo
PILE_DIR=/path/to/pile
PREPROCESSED_PILE_DIR=/path/to/preprocessed  # will be created by scripts/run_filter_domains.sh
MODEL_OUTPUT_DIR=/path/to/model_output_dir
PARTITION=partition # for slurm
mkdir -p ${CACHE}
mkdir -p ${MODEL_OUTPUT_DIR}
source ${DOREMI_DIR}/venv/bin/activate  # if you installed doremi in venv
```

Here is how to run the sample script for data preprocessing on The Pile, which separates the Pile data into domains and tokenizes it:
```
bash scripts/run_filter_domains.sh
```
Here is how to run baseline and DoReMi 280M models on preprocessed Pile data (tested on one node with 8 A100 GPUs):
```
bash scripts/runs/run_pile_baseline280M.sh
```
To run evaluation on a validation split, append `eval` to the end of the training script (`bash scripts/runs/run_pile_baseline280M.sh eval`).
After training a baseline model, we can run DoReMi:
```
bash scripts/runs/run_pile_doremi280M.sh
```
These scripts run for 200k steps, following the paper. The DoReMi run outputs domain weights in the `configs` directory with filename `<RUN_NAME>.json`. Note: so far, DoReMi has not been tested with gradient accumulation (although the code runs). If we accumulate the gradients for `k` steps, there will be `k-1` gradients computed against stale domain weights from the previous iteration (this problem doesn't exist for `k=1`).

## Running DoReMi on your own dataset
To run DoReMi on your own dataset, provide preprocessed (tokenized) data in the following format:
```
top_level/
    domain_name_1/
        files...
    domain_name_2/
        files...
    ...
```
where each inner directory (e.g., `domain_name_1`) can be loaded via HuggingFace's `load_from_disk` method. If your data is in a different format, you can add a custom data loading function in `doremi/dataloader.py`.
You will also need to write a config file and save it to `configs/` and write run scripts similar to `scripts/runs/run_pile_baseline280M.sh` and `scripts/runs/run_pile_doremi280M.sh` which refer to the config file. The config file specifies the mapping from domain name to mixture weight. The names do not have to be in order (DoReMi will always sort the domain names first to determine a fixed ordering) and the weights do not have to be normalized.
 
If this was useful to you, please cite the [paper](https://arxiv.org/abs/2305.10429):
```
@article{xie2023doremi,
  author = {Sang Michael Xie and Hieu Pham and Xuanyi Dong and Nan Du and Hanxiao Liu and Yifeng Lu and Percy Liang and Quoc V. Le and Tengyu Ma and Adams Wei Yu},
  journal = {arXiv preprint arXiv:2305.10429},
  title = {DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining},
  year = {2023},
}
```
