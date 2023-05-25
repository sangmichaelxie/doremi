#!/bin/bash

# python scripts/write_config.py --config_name baseline --preprocessed_dir /path/to/pile/data --cache_dir /path/to/cache
python scripts/write_config.py --config_name uniform
python scripts/write_config.py --config_name doremi_280M

