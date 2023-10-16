from pathlib import Path
from collections import Counter
import pickle
import random
from copy import deepcopy
from multiprocessing import Array
from itertools import cycle, chain
from functools import partial
import uuid
import numpy as np
import torch
from datasets import load_dataset, Dataset, IterableDataset
from datasets.info import DatasetInfo
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


RANDOM_BATCH_SIZE = 8192
DEFAULT_SEED=111

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
    def _iter_random_indices(rng, num_sources, probabilities_file=None, probabilities=None, random_batch_size=RANDOM_BATCH_SIZE):
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


def simulate_data_skip_per_domain(num_skip_examples, probabilities, rng, random_batch_size=RANDOM_BATCH_SIZE):
    num_sources = len(probabilities)
    sampled_domain_idxs = [
        rng.choice(num_sources, size=random_batch_size, p=probabilities)
        for _ in range(num_skip_examples // random_batch_size + 1)]
    sampled_domain_idxs = np.concatenate(sampled_domain_idxs)[:num_skip_examples]
    counts = Counter(sampled_domain_idxs)
    return [counts.get(i, 0) for i in range(num_sources)]


def determine_skip_per_domain(num_skip_examples, seed, domain_weights, domain_names):
    if num_skip_examples == 0:
        return {name: 0 for name in domain_names}

    if domain_weights is None or seed is None or domain_names is None:
        raise ValueError("If num_skip_examples > 0 then domain_weights, domain_names, and seed must not be None")

    rng = np.random.default_rng(seed)
    print('num_skip_examples')
    print(num_skip_examples)
    skip_per_domain = simulate_data_skip_per_domain(num_skip_examples, domain_weights, rng)
    domain_name_to_skip_num = {name: num for name, num in zip(domain_names, skip_per_domain)}
    return domain_name_to_skip_num


def skippable_data_gen(shards, num_skip_examples=0, loop=True, seed=111, shuffle=False):

    def get_shard_ds(shard_dir, num_skipped, seed, shuffle):
        shard = load_from_disk(dataset_path=str(shard_dir))
        if shuffle:
            shard = shard.shuffle(seed=seed)
        if num_skipped < num_skip_examples:
            # try to skip examples
            if len(shard) < (num_skip_examples - num_skipped):
                num_skipped += len(shard)
            else:
                shard = shard.select(range(num_skip_examples - num_skipped, len(shard)))
                logger.info(f"Skipped {num_skip_examples} examples in {shard_dir}")
                num_skipped = num_skip_examples
        return shard, num_skipped

    num_skipped = 0
    if loop:
        while True:
            for shard_dir in shards:
                shard, num_skipped = get_shard_ds(shard_dir, num_skipped, seed, shuffle)
                if num_skipped < num_skip_examples:
                    continue

                for ex in shard:
                    yield ex
                seed += 1
    else:
        for shard_dir in shards:
            shard, num_skipped = get_shard_ds(shard_dir, num_skipped, seed, shuffle)
            if num_skipped < num_skip_examples:
                continue

            for ex in shard:
                yield ex
            seed += 1


def get_pile_datasets(
        preprocessed_dir,
        cache_dir=None,
        split='train',
        seed=DEFAULT_SEED,
        domain_weights=None,
        domain_names=None,
        num_skip_examples=0,
        shuffle=False,
        shard_reversal=False):

    domain_name_to_skip_num = determine_skip_per_domain(num_skip_examples, seed, domain_weights, domain_names)

    print("domain_name_to_skip_num")
    print(domain_name_to_skip_num)

    preprocessed_dir = Path(preprocessed_dir) / split

    all_ds = {}
    for domain_dir in preprocessed_dir.iterdir():
        if split == 'train':
            shards = list(domain_dir.iterdir())
            if shard_reversal:
                curr_shards = list(reversed(shards))
            random.Random(seed).shuffle(shards)
        else:
            shards = [domain_dir]
        ds = IterableDataset.from_generator(
                skippable_data_gen,
                gen_kwargs={'shards': shards,
                            'num_skip_examples': domain_name_to_skip_num[domain_dir.name],
                            'loop': (split == 'train'),
                            'seed': seed,
                            'shuffle': shuffle}
                )
        all_ds[domain_dir.name] = ds
        seed += 1
    return all_ds


def get_perdomain_datasets(
        preprocessed_dir,
        domain_weights_dict,
        cache_dir=None,
        split=None,
        seed=DEFAULT_SEED,
        domain_weights=None,
        domain_names=None,
        num_skip_examples=0,
        shuffle=False,
        shard_reversal=False):
    '''
    Returns a dictionary from domain name to IterableDataset.
    '''
    domain_name_to_skip_num = determine_skip_per_domain(num_skip_examples, seed, domain_weights, domain_names)

    preprocessed_dir = Path(preprocessed_dir)
    if split is not None and (preprocessed_dir / split).exists():
        preprocessed_dir = preprocessed_dir / split
    else:
        logger.warn(f"No split used or split directory not found: using same data for all splits.")

    domains = list(sorted(domain_weights_dict.keys()))

    all_ds = {}
    for domain in domains:
        domain_dir = preprocessed_dir / domain

        if (domain_dir / 'dataset_info.json').exists():
            ds = load_from_disk(dataset_path=str(domain_dir))
            logger.info(f"Loaded {domain_dir}. Length: {len(ds)}")
        else:
            curr_shards = list(domain_dir.iterdir())
            if shard_reversal:
                curr_shards = list(reversed(curr_shards))
            # shuffle shard order
            random.Random(seed).shuffle(curr_shards)
            ds = IterableDataset.from_generator(
                    skippable_data_gen,
                    gen_kwargs={'shards': curr_shards,
                                'num_skip_examples': domain_name_to_skip_num[domain],
                                'loop': (split == 'train'),
                                'seed': seed,
                                'shuffle': shuffle}
                    )
            seed += 1
        all_ds[domain] = ds
    return all_ds


def get_preprocessed_mixed_dataset(
        preprocessed_dir,
        domain_weights_dict,
        dataset_name='pile',
        cache_dir=None,
        split='train',
        seed=DEFAULT_SEED,
        max_samples=None,
        add_domain_id=False,
        tmp_file=None,
        tokenizer=None,
        no_interleave=False,
        shuffle=False,
        num_skip_examples=0,
        shard_reversal=False):
    '''preprocessed_dir: has the following format
               first level: domain directories
               second level: shards for each domain. number of shards per domain should be the same.

       domain_weights_dict: dict from domain name to weight
       dataset_name: name of dataset. update this function to introduce new datasets
       cache_dir: cache directory for arrow files (if needed)
       split: train or validation
       seed: int (controls ordering of data shards)
       max_samples: int (limit for number of examples)
       add_domain_id: add domain id to the bath on the fly
       tmp_file: filename for saving domain weights to disk during doremi (otherwise, we keep the domain weights as a buffer saved along with the model)
       tokenizer: huggingface tokenizer
       no_interleave: don't interleave the domains - just iterate through the data in order
       shuffle: on-the-fly shuffle with a buffer size 100k
       num_skip_examples: skip examples in the iterator
       shard_reversal: reverse the shard ordering to prioritize unseen examples
    '''
    domain_names = list(sorted(domain_weights_dict.keys()))
    domain_to_idx = {domain_names[i]: i for i in range(len(domain_names))}
    domain_weights = np.asarray([domain_weights_dict[domain_name] for domain_name in domain_names])
    domain_weights = domain_weights / domain_weights.sum()

    # write domain weights to file if tmp_file is set
    if tmp_file is not None:
        probabilities_tmp_file = tmp_file

        with open(str(probabilities_tmp_file), 'wb') as f:
            pickle.dump(domain_weights, f)
        probabilities = None
    else:
        probabilities = domain_weights
        probabilities_tmp_file = None

    if dataset_name == 'pile':
        all_ds = get_pile_datasets(
                preprocessed_dir,
                cache_dir=cache_dir,
                split=split,
                seed=seed,
                domain_weights=domain_weights,
                domain_names=domain_names,
                num_skip_examples=num_skip_examples,
                shuffle=shuffle,
                shard_reversal=shard_reversal)
    else:
        try:
            all_ds = get_perdomain_datasets(
                preprocessed_dir, 
                domain_weights_dict,
                cache_dir=cache_dir,
                split=split,
                seed=seed,
                domain_weights=domain_weights,
                domain_names=domain_names,
                num_skip_examples=num_skip_examples,
                shuffle=shuffle,
                shard_reversal=shard_reversal)
        except Exception:
            raise ValueError(f"dataset_name {dataset_name} not implemented.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def add_domain_id_generator(ds, domain_idx):
        for ex in ds:
            ex['domain_id'] = domain_idx
            yield ex

    domain_ds_ls = []
    for domain_name in domain_names:
        domain_idx = domain_to_idx[domain_name]
        domain_ds = all_ds[domain_name]
        # add domain_id if necessary
        if add_domain_id:
            domain_ds = IterableDataset.from_generator(add_domain_id_generator, gen_kwargs={'ds': domain_ds, 'domain_idx': domain_idx})
        domain_ds_ls.append(domain_ds)

    if no_interleave:
        # instead of interleaving, run through each dataset
        def data_generator(shards):
            for shard in shards:
                for ex in shard:
                    yield ex
        ds = IterableDataset.from_generator(data_generator, gen_kwargs={'shards': domain_ds_ls})
        logger.info("Not interleaving dataset - will not sample according to domain weights")

    else:
        ds = interleave_datasets(
                domain_ds_ls,
                probabilities=probabilities,
                probabilities_file=probabilities_tmp_file,
                seed=seed)

    def take_data_generator(ds, max_samples):
        idx = 0
        for ex in ds:
            yield ex
            idx += 1
            if max_samples is not None and idx >= max_samples:
                return

    ds = IterableDataset.from_generator(take_data_generator, gen_kwargs={'ds': ds, 'max_samples': max_samples})
    return ds


def get_data_collator(tokenizer, return_tensors='pt', do_padding=False):
    def data_collator(features):
        if not do_padding:
            batch = {
                    k: torch.tensor([f[k] for f in features])
                    for k in features[0].keys()
                    }
        else:
            batch = tokenizer.pad(features, return_tensors=return_tensors, pad_to_multiple_of=tokenizer.model_max_length)
        batch['attention_mask'] = batch['attention_mask'].long()
        batch['input_ids'] = batch['input_ids'].long()

        batch.pop("special_tokens_mask", None)
        if 'labels' not in batch:
            labels = batch['input_ids'].clone()
            batch["labels"] = labels

        if tokenizer.pad_token_id is not None:
            batch['labels'][batch['labels'] == tokenizer.pad_token_id] = -100

        if 'domain_ids' not in batch and 'domain_id' in batch:
            batch['domain_ids'] = batch['domain_id']  # compat
            batch.pop('domain_id')
        return batch
    return data_collator
