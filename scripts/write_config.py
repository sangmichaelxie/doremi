from pathlib import Path
import json
import argparse
from datasets import load_from_disk
from collections import defaultdict
import shutil
import math
from tqdm import tqdm
from joblib import Parallel, delayed
from transformers import AutoTokenizer


PILE_DOMAINS = ['ArXiv', 'BookCorpus2', 'Books3', 'DM Mathematics', 'Enron Emails', 'EuroParl', 'FreeLaw', 'Github', 'Gutenberg (PG-19)', 'HackerNews', 'NIH ExPorter', 'OpenSubtitles', 'OpenWebText2', 'PhilPapers', 'Pile-CC', 'PubMed Abstracts', 'PubMed Central', 'StackExchange', 'USPTO Backgrounds', 'Ubuntu IRC', 'Wikipedia (en)', 'YoutubeSubtitles']


def compute_pile_baseline_weights(preprocessed_dir, cache_dir, nopack=False, tokenizer=None):

    preprocessed_dir = Path(preprocessed_dir) / 'train'

    def process_shard(shard_dir):
        curr_count = 0
        ds = load_from_disk(dataset_path=str(shard_dir))
        if nopack:
            # in the DoReMi paper, we first padded to the context length then counted
            # the number of chunks, and dynamically packed the examples
            # together (possibly even from different domains)
            num_tokens_in_curr_doc = 0
            chunk_size = 1024
            for ex in tqdm(ds):
                toks = ex['input_ids']
                sep_idxs = [i for i in range(len(toks)) if toks[i] == tokenizer.eos_token_id]
                if len(sep_idxs) > 0:
                    prev_sep_idx = -1
                    for sep_idx in sep_idxs:
                        num_tokens_in_curr_doc += sep_idx - prev_sep_idx - 1
                        prev_sep_idx = sep_idx
                        curr_count += math.ceil(num_tokens_in_curr_doc / chunk_size)
                        num_tokens_in_curr_doc = 0
                    if prev_sep_idx != len(toks) - 1:
                        num_tokens_in_curr_doc += len(toks) - prev_sep_idx - 1
                else:
                    num_tokens_in_curr_doc += len(toks)
            if num_tokens_in_curr_doc > 0:
                curr_count += math.ceil(num_tokens_in_curr_doc / chunk_size)
        else:
            curr_count = len(ds)

        return curr_count

    domain_lens = defaultdict(int)
    for domain_dir in preprocessed_dir.iterdir():
        print("Counting domain", domain_dir.name)
        counts = Parallel(n_jobs=30)(delayed(process_shard)(shard_dir) for shard_dir in domain_dir.iterdir())
        domain_lens[domain_dir.name] = sum(counts)

    # multiply by epochs to get weights according to effective sizes
    pile_epochs = {
        'Pile-CC': 1.0,
        'PubMed Central': 2.0,
        'Books3': 1.5,
        'OpenWebText2': 2.0,
        'ArXiv': 2.0,
        'Github': 1.0,
        'FreeLaw': 1.5,
        'StackExchange': 2.0,
        'USPTO Backgrounds': 2.0,
        'PubMed Abstracts': 2.0,
        'Gutenberg (PG-19)': 2.5,
        'OpenSubtitles': 1.5,
        'Wikipedia (en)': 3.0,
        'DM Mathematics': 2.0,
        'Ubuntu IRC': 2.0,
        'BookCorpus2': 1.5,
        'EuroParl': 2.0,
        'HackerNews': 2.0,
        'YoutubeSubtitles': 2.0,
        'PhilPapers': 2.0,
        'NIH ExPorter': 2.0,
        'Enron Emails': 2.0}
    domain_lens = {k: v * pile_epochs[k] for k, v in domain_lens.items()}

    # renormalize domain_lens
    total_len = sum(domain_lens.values())
    domain_lens = {k: v / total_len for k, v in domain_lens.items()}
    print("Baseline domain weights:", domain_lens)
    return domain_lens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="baseline")
    parser.add_argument("--preprocessed_dir", type=str, default="/path/to/preprocessed")
    parser.add_argument("--cache_dir", type=str, default="/path/to/cache")
    parser.add_argument("--nopack", action='store_true')
    parser.add_argument("--tokenizer", type=str, default='togethercomputer/RedPajama-INCITE-Base-7B-v0.1')
    args = parser.parse_args()

    config_dir = Path("configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{args.config_name}.json"

    if args.config_name.startswith('pile_baseline') and args.config_name != 'pile_baseline_256kvocab':
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        domain_weights = compute_pile_baseline_weights(args.preprocessed_dir, args.cache_dir, nopack=args.nopack, tokenizer=tokenizer)
        config = {
            "train_domain_weights": domain_weights,
            "eval_domain_weights": domain_weights,
            }
    elif args.config_name == 'pile_uniform':
        domain_weights = {d: 1 / len(PILE_DOMAINS) for d in PILE_DOMAINS}
        config = {
            "train_domain_weights": domain_weights,
            "eval_domain_weights": domain_weights,
            }
    elif args.config_name == 'pile_baseline_256kvocab':
        domain_weights = {
                "OpenWebText2": 0.1247,
                "USPTO Backgrounds": 0.0420,
                "NIH ExPorter": 0.0052,
                "Wikipedia (en)": 0.0919,
                "YoutubeSubtitles": 0.0042,
                "Books3": 0.0676,
                "HackerNews": 0.0075,
                "StackExchange": 0.0929,
                "Enron Emails": 0.0030,
                "FreeLaw": 0.0386,
                "DM Mathematics": 0.0198,
                "PubMed Central": 0.1071,
                "OpenSubtitles": 0.0124,
                "BookCorpus2": 0.0044,
                "Ubuntu IRC": 0.0074,
                "PhilPapers": 0.0027,
                "PubMed Abstracts": 0.0845,
                "EuroParl": 0.0043,
                "Github": 0.0427,
                "Gutenberg (PG-19)": 0.0199,
                "Pile-CC": 0.1121,
                "ArXiv": 0.1052
                }
        config = {
            "train_domain_weights": domain_weights,
            "eval_domain_weights": domain_weights,
            }
    elif args.config_name == 'pile_doremi_280M_256kvocab':
        domain_weights = {
                'Pile-CC': 0.6057,
                'PubMed Central': 0.0046,
                'Books3': 0.0224,
                'OpenWebText2': 0.1019,
                'ArXiv': 0.0036,
                'Github': 0.0179,
                'FreeLaw': 0.0043,
                'StackExchange': 0.0153,
                'USPTO Backgrounds': 0.0036,
                'PubMed Abstracts': 0.0113,
                'Gutenberg (PG-19)': 0.0072,
                'OpenSubtitles': 0.0047,
                'Wikipedia (en)': 0.0699,
                'DM Mathematics': 0.0018,
                'Ubuntu IRC': 0.0093,
                'BookCorpus2': 0.0061,
                'EuroParl': 0.0062,
                'HackerNews': 0.0134,
                'YoutubeSubtitles': 0.0502,
                'PhilPapers': 0.0274,
                'NIH ExPorter': 0.0063,
                'Enron Emails': 0.0070}

        config = {
            "train_domain_weights": domain_weights,
            "eval_domain_weights": domain_weights,
            }
    elif args.config_name == 'rp_baseline':
        domain_weights = {
                'common_crawl': 0.7316,
                'c4': 0.1458,
                'github': 0.0492,
                'wikipedia': 0.02,
                'book': 0.0216,
                'arxiv': 0.0233,
                'stackexchange': 0.016,
        }
        config = {
            "train_domain_weights": domain_weights,
            "eval_domain_weights": domain_weights,
            }
    else:
        raise ValueError(f"Unknown config name {args.config_name}")

    print(json.dumps(config, indent=2))
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == '__main__':
    main()
