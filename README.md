# DoReMi: Domain Reweighting with Minimax Optimization

PyTorch reference implementation of DoReMi, a method for optimizing data mixtures for language modeling datasets. Check out the [paper](https://arxiv.org/abs/2305.10429) for more details. This repo is currently in active development!

Note that there may be a few differences between this repo and the paper, namely:
- PyTorch vs JAX
- Subtle differences in model architecture
- Tokenizers used (256k vocab size used in paper, while standard GPT2 tokenizer is 50k vocab size). This can siginificantly affect the data mixtures as calculated by token count.

To get started, please clone the repo, and install it:
```
git clone git@github.com:/sangmichaelxie/doremi.git
pip install -e doremi
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
Here is how to run baseline and DoReMi 280M models on preprocessed Pile data:
```
bash scripts/runs/run_pile_baseline280M.sh
```
After training a baseline model, we can run DoReMi:
```
bash scripts/runs/run_pile_doremi280M.sh
```

If this was useful to you, please cite the [paper](https://arxiv.org/abs/2305.10429):
```
@article{xie2023doremi,
  author = {Sang Michael Xie and Hieu Pham and Xuanyi Dong and Nan Du and Hanxiao Liu and Yifeng Lu and Percy Liang and Quoc V. Le and Tengyu Ma and Adams Wei Yu},
  journal = {arXiv preprint arXiv:2305.10429},
  title = {DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining},
  year = {2023},
}
```
