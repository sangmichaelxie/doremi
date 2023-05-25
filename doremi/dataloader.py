from pathlib import Path
import pickle
from copy import deepcopy
from multiprocessing import Array
from itertools import cycle
import uuid
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from datasets import load_dataset, Dataset, IterableDataset
from datasets.iterable_dataset import ExamplesIterable, RandomlyCyclingMultiSourcesExamplesIterable
from transformers import AutoTokenizer, default_data_collator
import torch.distributed as dist
from tqdm import tqdm
import math
from datasets.filesystems import _reset_fsspec_lock
from datasets.utils.logging import get_logger
from datasets import load_from_disk
import shutil

logger = get_logger(__name__)


PILE_DOMAINS = ['ArXiv', 'BookCorpus2', 'Books3', 'DM Mathematics', 'Enron Emails', 'EuroParl', 'FreeLaw', 'Github', 'Gutenberg (PG-19)', 'HackerNews', 'NIH ExPorter', 'OpenSubtitles', 'OpenWebText2', 'PhilPapers', 'Pile-CC', 'PubMed Abstracts', 'PubMed Central', 'StackExchange', 'USPTO Backgrounds', 'Ubuntu IRC', 'Wikipedia (en)', 'YoutubeSubtitles']

DOMAIN_TO_IDX = {
    name: idx for idx, name in enumerate(PILE_DOMAINS)}

PILE_SUBSETS = [f'0{i}' if i < 10 else str(i) for i in range(0, 30)]


class UpdatableRandomlyCyclingMultiSourcesExamplesIterable(
        RandomlyCyclingMultiSourcesExamplesIterable):

    def __init__(self, ex_iterables, generator, probabilities=None, probabilities_file=None, stopping_strategy="all_exhausted"):
        '''
        probabilities: vector of static probabilities over training
        probabilities_file: tmp file to store dynamically changing probabilities
        '''
        super().__init__(ex_iterables, generator, stopping_strategy=stopping_strategy)
        self.probabilities_file = probabilities_file
        self.probabilities = probabilities

    @staticmethod
    def _iter_random_indices(rng, num_sources, probabilities_file=None, probabilities=None, random_batch_size=8192):
        while True:
            # read domain weights
            if probabilities_file is not None:
                with open(probabilities_file, 'rb') as f:
                    probabilities = pickle.load(f)

            yield from (int(i) for i in rng.choice(num_sources, size=random_batch_size, p=probabilities))

    def _give_indice_iterator(self):
        rng = deepcopy(self.generator)
        return self._iter_random_indices(rng, len(self.ex_iterables), probabilities_file=self.probabilities_file, probabilities=self.probabilities)

    def shard_data_sources(self, shard_indices):
        return self

    @property
    def n_shards(self):
        return 1

    def shuffle_data_sources(self, seed):
        self.ex_iterables = [ex_iterable.shuffle_data_sources(seed) for ex_iterable in self.ex_iterables]
        return self


def interleave_datasets(datasets, probabilities=None, probabilities_file=None, seed=None, stopping_strategy='all_exhausted'):
    iterable_datasets = []
    for dataset in datasets:
        if not isinstance(dataset, IterableDataset):
            iterable_datasets.append(dataset.to_iterable_dataset())
        else:
            iterable_datasets.append(dataset)

    ex_iterables = [d._ex_iterable for d in iterable_datasets]

    generator = np.random.default_rng(seed)
    ex_iterable = UpdatableRandomlyCyclingMultiSourcesExamplesIterable(
            ex_iterables, generator=generator,
            probabilities=probabilities, probabilities_file=probabilities_file,
            stopping_strategy=stopping_strategy)

    return IterableDataset(ex_iterable=ex_iterable)


def get_dataset(pile_dir, domain_ids_dir, cache_dir=None, split='train'):
    # initialize streaming datasets from pile_dir
    pile_dir = Path(pile_dir)
    domain_ids_split_dir = Path(domain_ids_dir) / split
    if split == 'train':
        data_files = [str(pile_dir / f"{subset}.jsonl.zst") for subset in PILE_SUBSETS]
        domain_ids = [np.load(domain_ids_split_dir / f"{subset}_domain_ids.npy") for subset in PILE_SUBSETS]
    elif split == 'validation':
        data_files = [str(pile_dir / "val.jsonl.zst")]
        domain_ids = [np.load(domain_ids_split_dir / f"{split}_domain_ids.npy")]
    else:
        data_files = [str(pile_dir / f"test.jsonl.zst")]
        domain_ids = [np.load(domain_ids_split_dir / f"{split}_domain_ids.npy")]


    ds_ls = []
    for data_file in data_files:
        ds = load_dataset('json',
                          data_files=[data_file],
                          cache_dir=cache_dir,
                          streaming=True)['train']
        ds_ls.append(ds)
    return ds_ls, domain_ids


def pile_transform(tokenizer, max_length, seed=None):
    def transform(batch, domain_id):
        def calculate_chunk_starts(tokens):
            num_chunks = max(1, len(tokens) // max_length)
            return np.arange(0, len(tokens), len(tokens) // num_chunks)

        # tokenize
        inputs = tokenizer(batch['text'])
        # chunk them and add domain id
        total_length = len(inputs['input_ids'])
        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length
            return [{**{k: v[i:i+max_length] for k, v in inputs.items()}, **{'domain_ids': domain_id}}
                      for i in range(0, total_length, max_length)]
        else:
            for k, v in inputs.items():
                if k == 'input_ids':
                    inputs[k] = v + [tokenizer.pad_token_id] * (max_length - total_length)
                else:
                    inputs[k] = v + [0] * (max_length - total_length)
            return [{**inputs, **{'domain_ids': domain_id}}]

    return transform


def get_preprocessed_mixed_dataset(preprocessed_dir, domain_weights_dict, cache_dir=None, split='train', sharded=True, seed=None, filter_domains_fn=None, max_samples=None, tmp_file=None):
    '''preprocessed_dir: has the following format
               first level: domain directories
               second level: shards for each domain. number of shards per domain should be the same.

       domain_weights_dict: dict from domain name to weight
    '''

    preprocessed_dir = Path(preprocessed_dir)
    cached_preprocessed_dir = Path(cache_dir) / 'preprocessed_cache' / preprocessed_dir.name
    cached_preprocessed_dir.parent.mkdir(parents=True, exist_ok=True)

    if not cached_preprocessed_dir.exists():
        print("Copying preprocessed files to cache")
        shutil.copytree(str(preprocessed_dir), str(cached_preprocessed_dir))

    def iterdir_with_filter(dir_path):
        for f in dir_path.iterdir():
            if filter_domains_fn is None or filter_domains_fn(f):
                yield f

    preprocessed_dir = cached_preprocessed_dir / split
    first_domain_dir = list(preprocessed_dir.iterdir())[0]
    if sharded:
        num_shards = len(list(iterdir_with_filter(first_domain_dir)))
    else:
        num_shards = 1

    all_ds_shards = [{} for _ in range(num_shards)]
    for domain_dir in preprocessed_dir.iterdir():
        domain_shard_ds_ls = []
        if sharded:
            for shard_idx, shard_dir in enumerate(iterdir_with_filter(domain_dir)):
                ds = load_from_disk(dataset_path=str(shard_dir))
                all_ds_shards[shard_idx][domain_dir.name] = ds
        else:
            all_ds_shards[0][domain_dir.name] = load_from_disk(dataset_path=str(domain_dir))

    domain_names = list(sorted(domain_weights_dict.keys()))
    domain_weights = np.asarray([domain_weights_dict[domain_name] for domain_name in domain_names])
    domain_weights = domain_weights / domain_weights.sum()

    if tmp_file is not None:
        probabilities_tmp_file = tmp_file

        with open(str(probabilities_tmp_file), 'wb') as f:
            pickle.dump(domain_weights, f)
        probabilities = None
    else:
        probabilities = domain_weights
        probabilities_tmp_file = None

    per_domain_ds_shards = []
    for domain_ds_dict in all_ds_shards:
        domain_ds_ls = [domain_ds_dict[domain_name] for domain_name in domain_names]
        mixed_ds_shard = interleave_datasets(
                domain_ds_ls,
                probabilities=probabilities,
                probabilities_file=probabilities_tmp_file,
                seed=seed)
        per_domain_ds_shards.append(mixed_ds_shard)

    def data_generator(shards, max_samples=None):
        idx = 0
        for shard in shards:
            for ex in shard:
                yield ex
                idx += 1
                if max_samples is not None and idx >= max_samples:
                    return

    return IterableDataset.from_generator(data_generator, gen_kwargs={'shards': per_domain_ds_shards, 'max_samples': max_samples})


def get_data_collator(tokenizer, return_tensors='pt'):
    def data_collator(features):
        batch = {
                k: torch.tensor([f[k] for f in features])
                for k in features[0].keys()
                }
        batch['attention_mask'] = batch['attention_mask'].long()
        batch['input_ids'] = batch['input_ids'].long()

        batch.pop("special_tokens_mask", None)
        labels = batch['input_ids'].clone()
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        if 'domain_ids' not in batch and 'domain_id' in batch:
            batch['domain_ids'] = batch['domain_id']  # compat
            batch.pop('domain_id')
        return batch
    return data_collator



if __name__ == "__main__":
    # a short test

    domain_weights_dict = {domain: 1 for domain in PILE_DOMAINS}
    ds, domain_weights = get_preprocessed_mixed_dataset(
            preprocessed_dir='/path/to/preprocessed', # run filter_domains.py in scripts/
            domain_weights_dict=domain_weights_dict,
            cache_dir='/path/to/cache',
            split='train',
            sharded=True,
            filter_domains_fn=lambda f: f.name != '00')

    tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataloader = DataLoader(
            ds, batch_size=512, num_workers=1, collate_fn=get_data_collator(tokenizer))

    domain_weights_dict_2 = {domain: 1 if domain == 'Books3' else 0 for domain in PILE_DOMAINS}
    domain_weights_2_vec = torch.tensor(list(domain_weights_dict_2.values()))
    domain_weights_2_vec = domain_weights_2_vec / domain_weights_2_vec.sum()
    phase_1_domains = [0] * len(PILE_DOMAINS)
    phase_2_domains = [0] * len(PILE_DOMAINS)
    for i, batch in tqdm(enumerate(dataloader)):
        if i < 500:
            for domain_id in batch['domain_ids']:
                phase_1_domains[domain_id] += 1
        elif i < 1000:
            if i == 500:
                # dataloader.dataset._ex_iterable.set_domain_weights(domain_weights_2_vec)
                domain_weights[:] = domain_weights_2_vec[:]
            for domain_id in batch['domain_ids']:
                phase_2_domains[domain_id] += 1
        else:
            break

    phase_1_domains = np.asarray(phase_1_domains)
    phase_2_domains = np.asarray(phase_2_domains)
    print("Phase 1")
    print({domain: count / phase_1_domains.sum() for domain, count in zip(PILE_DOMAINS, phase_1_domains)})

    print("Phase 2")
    print({domain: count / phase_2_domains.sum() for domain, count in zip(PILE_DOMAINS, phase_2_domains)})





