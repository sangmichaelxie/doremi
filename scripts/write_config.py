from pathlib import Path
import json
import argparse
from datasets import load_from_disk
from collections import defaultdict
import shutil


PILE_DOMAINS = ['ArXiv', 'BookCorpus2', 'Books3', 'DM Mathematics', 'Enron Emails', 'EuroParl', 'FreeLaw', 'Github', 'Gutenberg (PG-19)', 'HackerNews', 'NIH ExPorter', 'OpenSubtitles', 'OpenWebText2', 'PhilPapers', 'Pile-CC', 'PubMed Abstracts', 'PubMed Central', 'StackExchange', 'USPTO Backgrounds', 'Ubuntu IRC', 'Wikipedia (en)', 'YoutubeSubtitles']


def compute_pile_baseline_weights(preprocessed_dir, cache_dir):

    def filter_domains_fn(f):
        return f.name != '00'

    def iterdir_with_filter(dir_path):
        for f in dir_path.iterdir():
            if filter_domains_fn is None or filter_domains_fn(f):
                yield f

    preprocessed_dir = Path(preprocessed_dir)
    cached_preprocessed_dir = Path(cache_dir) / 'preprocessed_cache' / preprocessed_dir.name
    cached_preprocessed_dir.parent.mkdir(parents=True, exist_ok=True)

    if not cached_preprocessed_dir.exists():
        print("Copying preprocessed files to cache")
        shutil.copytree(str(preprocessed_dir), str(cached_preprocessed_dir))

    preprocessed_dir = cached_preprocessed_dir / 'train'

    first_domain_dir = list(preprocessed_dir.iterdir())[0]
    num_shards = len(list(iterdir_with_filter(first_domain_dir)))

    domain_lens = defaultdict(int)
    for domain_dir in preprocessed_dir.iterdir():
        print("Counting domain", domain_dir.name)
        domain_shard_ds_ls = []
        for shard_idx, shard_dir in enumerate(iterdir_with_filter(domain_dir)):
            ds = load_from_disk(dataset_path=str(shard_dir))
            domain_lens[domain_dir.name] += len(ds)
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
    args = parser.parse_args()

    config_dir = Path("configs")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{args.config_name}.json"

    if args.config_name == 'baseline':
        domain_weights = compute_pile_baseline_weights(args.preprocessed_dir, args.cache_dir)
        config = {
            "train_domain_weights": domain_weights,
            "eval_domain_weights": domain_weights,
            }
    elif args.config_name == 'uniform':
        domain_weights = {d: 1 / len(PILE_DOMAINS) for d in PILE_DOMAINS}
        config = {
            "train_domain_weights": domain_weights,
            "eval_domain_weights": domain_weights,
            }
    elif args.config_name == 'doremi_280M':
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

    with open(config_path, 'w') as f:
        json.dump(config, f)

if __name__ == '__main__':
    main()
