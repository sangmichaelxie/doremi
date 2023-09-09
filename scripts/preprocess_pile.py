"""
Filter the Pile data into domains and tokenize the data.
"""
from datasets import load_dataset, Dataset, IterableDataset
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
from pathlib import Path
from itertools import cycle
import torch
import numpy as np
from datasets import Features, Sequence, Value
import shutil
from itertools import chain
from tokenizers.processors import TemplateProcessing


PILE_DOMAINS = ['ArXiv', 'BookCorpus2', 'Books3', 'DM Mathematics', 'Enron Emails', 'EuroParl', 'FreeLaw', 'Github', 'Gutenberg (PG-19)', 'HackerNews', 'NIH ExPorter', 'OpenSubtitles', 'OpenWebText2', 'PhilPapers', 'Pile-CC', 'PubMed Abstracts', 'PubMed Central', 'StackExchange', 'USPTO Backgrounds', 'Ubuntu IRC', 'Wikipedia (en)', 'YoutubeSubtitles']

DOMAIN_TO_IDX = {
    name: idx for idx, name in enumerate(PILE_DOMAINS)}

PILE_SUBSETS = [f'0{i}' if i < 10 else str(i) for i in range(0, 30)]


def pile_transform(tokenizer, max_length, seed=None):
    def transform(batch):
        # tokenize
        examples = tokenizer(batch['text'])

        # Concatenate all texts. attention mask is all 1
        examples = {k: list(chain(*examples[k])) for k in examples.keys() if k!= 'attention_mask'}
        total_length = len(examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_length:
            total_length = (total_length // max_length) * max_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in examples.items()
        }
        return result

    return transform



def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--pile_path_dir', type=str, default='/path/to/pile')
    parser.add_argument('--output_dir', type=str, default='/path/to/output')
    parser.add_argument('--intermediate_dir', type=str, default='/path/to/intermediate')
    parser.add_argument('--domain', type=str, default='Books3')
    parser.add_argument('--subset', type=str, default='01')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--nproc', type=int, default=8)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    parser.add_argument('--cache_dir', type=str, default='/path/to/cache')
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()

    args.domain = args.domain.replace('_', ' ')

    # move from intermediate dir to output dir
    if args.split == 'train':
        output_dir = Path(args.output_dir) / args.split / args.domain / args.subset
    else:
        output_dir = Path(args.output_dir) / args.split / args.domain
    if output_dir.exists():
        print("Already done, skipping")
        return

    pile_dir = Path(args.pile_path_dir)
    if args.split == 'train':
        data_files = [str(pile_dir / f"{args.subset}.jsonl.zst")]
    elif args.split == 'validation':
        data_files = [str(pile_dir / "val.jsonl.zst")]
    else:
        data_files = [str(pile_dir / f"test.jsonl.zst")]


    # load dataset
    ds = load_dataset('json',
                      data_files=data_files,
                      cache_dir=args.cache_dir,
                      streaming=True)['train']

    def filter_fn(ex, idx):
        return ex['meta']['pile_set_name'] == args.domain
    def domain_id_fn(ex):
        return DOMAIN_TO_IDX[args.domain]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    # add a separator token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
            single="$A "+tokenizer.eos_token,
            special_tokens=[(tokenizer.eos_token, tokenizer.eos_token_id)])
    transform = pile_transform(tokenizer, args.max_length, seed=args.seed)

    ds = ds.filter(filter_fn, with_indices=True)
    ds = ds.map(transform, batched=True, remove_columns=['text', 'meta'])

    # create a generator
    def data_generator():
        count = 0
        for i, ex in enumerate(ds):
            ex['domain_id'] = domain_id_fn(ex)
            yield ex
            count += 1

    features = Features({
            "input_ids": Sequence(Value("int32")),
            "domain_id": Value("int32"),
        })
    processed_ds = Dataset.from_generator(data_generator, features=features)

    # save dataset
    if args.split == 'train':
        intermediate_dir = Path(args.intermediate_dir) / args.split / args.domain / args.subset
    else:
        intermediate_dir = Path(args.intermediate_dir) / args.split / args.domain

    intermediate_dir.parent.mkdir(parents=True, exist_ok=True)
    processed_ds.save_to_disk(intermediate_dir, max_shard_size='1GB', num_proc=args.nproc)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(intermediate_dir), str(output_dir))


if __name__ == '__main__':
    main()
