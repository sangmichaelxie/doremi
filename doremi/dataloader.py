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
from torch.utils.data import DataLoader, Sampler
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
    skip_per_domain = simulate_data_skip_per_domain(num_skip_examples, domain_weights, rng)
    domain_name_to_skip_num = {name: num for name, num in zip(domain_names, skip_per_domain)}
    return domain_name_to_skip_num


def skippable_data_gen(shards, num_skip_examples=0):
    num_skipped = 0
    while True:
        for shard_dir in shards:
            shard = load_from_disk(dataset_path=str(shard_dir))
            if num_skipped < num_skip_examples:
                # try to skip examples
                if len(shard) < (num_skip_examples - num_skipped):
                    num_skipped += len(shard)
                    continue
                else:
                    shard = shard.select(range(num_skip_examples - num_skipped, len(shard)))
                    logger.info(f"Skipped {num_skip_examples} examples in {shard_dir}")
                    num_skipped = num_skip_examples

            for ex in shard:
                yield ex


def get_pile_datasets(
        preprocessed_dir,
        cache_dir=None,
        split='train',
        seed=DEFAULT_SEED,
        domain_weights=None,
        domain_names=None,
        num_skip_examples=0):

    domain_name_to_skip_num = determine_skip_per_domain(num_skip_examples, seed, domain_weights, domain_names)

    preprocessed_dir = Path(preprocessed_dir) / split

    all_ds = {}
    for domain_dir in preprocessed_dir.iterdir():
        if split == 'train':
            shards = list(domain_dir.iterdir())
            random.Random(seed).shuffle(shards)
        else:
            shards = [domain_dir]
        ds = IterableDataset.from_generator(
                skippable_data_gen,
                gen_kwargs={'shards': shards,
                            'num_skip_examples': domain_name_to_skip_num[domain_dir.name]}
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
        num_skip_examples=0):
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
            # shuffle shard order
            random.Random(seed).shuffle(curr_shards)
            ds = IterableDataset.from_generator(
                    skippable_data_gen,
                    gen_kwargs={'shards': curr_shards,
                                'num_skip_examples': domain_name_to_skip_num[domain]}
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
        num_skip_examples=0):
    '''preprocessed_dir: has the following format
               first level: domain directories
               second level: shards for each domain. number of shards per domain should be the same.

       domain_weights_dict: dict from domain name to weight
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
                num_skip_examples=num_skip_examples)
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
                num_skip_examples=num_skip_examples)
        except Exception:
            raise ValueError(f"dataset_name {dataset_name} not implemented.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def add_domain_id_fn(example, domain_idx):
        if 'domain_id' not in example:
            example['domain_id'] = domain_idx
        return example

    domain_ds_ls = []
    for domain_name in domain_names:
        domain_idx = domain_to_idx[domain_name]
        domain_ds = all_ds[domain_name]
        # add domain_id if necessary
        if add_domain_id:
            domain_ds = domain_ds.map(partial(add_domain_id_fn, domain_idx=domain_idx))
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
    if shuffle:
        ds = ds.shuffle(seed=seed+2, buffer_size=10000)
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



if __name__ == "__main__":
    # a short test

    PILE_DOMAINS = ['ArXiv', 'BookCorpus2', 'Books3', 'DM Mathematics', 'Enron Emails', 'EuroParl', 'FreeLaw', 'Github', 'Gutenberg (PG-19)', 'HackerNews', 'NIH ExPorter', 'OpenSubtitles', 'OpenWebText2', 'PhilPapers', 'Pile-CC', 'PubMed Abstracts', 'PubMed Central', 'StackExchange', 'USPTO Backgrounds', 'Ubuntu IRC', 'Wikipedia (en)', 'YoutubeSubtitles']

    DOMAIN_TO_IDX = {
        name: idx for idx, name in enumerate(PILE_DOMAINS)}

    PILE_SUBSETS = [f'0{i}' if i < 10 else str(i) for i in range(0, 30)]

    domain_weights_dict = {domain: 1 for domain in PILE_DOMAINS}
    ds, domain_weights = get_preprocessed_mixed_dataset(
            preprocessed_dir='/path/to/preprocessed', # run filter_domains.py in scripts/
            domain_weights_dict=domain_weights_dict,
            cache_dir='/path/to/cache',
            split='train',
            sharded=True)

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





