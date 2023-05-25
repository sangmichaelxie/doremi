# DoReMi: Domain Reweighting with Minimax Optimization

PyTorch reference implementation of DoReMi, a method for optimizing data mixtures for language modeling datasets. This repo is currently in active development!

To get started, please clone the repo, and install it:
```
git clone git@github.com:/sangmichaelxie/doremi
pip install -e doremi
```

All code should be run from the `doremi` directory. You'll need to edit the scripts below to run on your own paths to Pile data, cache directories, etc.
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
