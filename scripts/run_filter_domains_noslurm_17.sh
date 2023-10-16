# Sample commands to run Pile data preprocessing

# load global parameters
source constants.sh

INTERMEDIATE_SCRATCH_PATH=${CACHE}/pile_preprocessed_tmp

SPLIT=train
for PILE_DOMAIN in "Ubuntu_IRC"; do
for SUBSET in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29; do
LOGDIR=logs/filter_domain_logs/${SPLIT}
mkdir -p ${LOGDIR}
echo "Processing ${PILE_DOMAIN}"
python scripts/filter_domains.py --pile_path_dir ${PILE_DIR} --output_dir ${PREPROCESSED_PILE_DIR} --intermediate_dir ${INTERMEDIATE_SCRATCH_PATH} --cache_dir ${CACHE} --split ${SPLIT} --domain ${PILE_DOMAIN} --num_samples 102400000 --tokenizer gpt2 --seed 111 --nproc 1 --subset ${SUBSET}

echo "Done with ${PILE_DOMAIN}"
done
done
