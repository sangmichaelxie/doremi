# Sample code to run Pile data preprocessing on 1/3 of the Pile, keeping 00 held out

PILE_PATH=/path/to/pile
PREPROCESSED_OUTPUT_PATH=/path/to/preprocessed
INTERMEDIATE_SCRATCH_PATH=/path/to/scratch
CACHE=/path/to/cache
PARTITION=partition


SPLIT=train
for PILE_DOMAIN in "ArXiv" "DM_Mathematics" "Enron_Emails" "EuroParl" "FreeLaw" "Github" "HackerNews" "NIH_ExPorter" "OpenSubtitles" "OpenWebText2" "PhilPapers" "Pile-CC" "PubMed_Abstracts" "PubMed_Central" "StackExchange" "USPTO_Backgrounds" "Wikipedia_(en)" "YoutubeSubtitles"; do
for SUBSET in 00 01 02 03 04 05 06 07 08 09 10; do
LOGDIR=logs/filter_domain_logs/${SPLIT}
mkdir -p ${LOGDIR}
jid=$(sbatch \
        --parsable \
        --partition ${PARTITION} \
        --mem 8G \
        -c 1 \
        --output ${LOGDIR}/${PILE_DOMAIN}_${SUBSET} \
        scripts/run.sh "python scripts/filter_domains.py --pile_path_dir ${PILE_PATH} --output_dir ${PREPROCESSED_OUTPUT_PATH} --intermediate_dir ${INTERMEDIATE_SCRATCH_PATH} --cache_dir ${CACHE} --split ${SPLIT} --domain \"${PILE_DOMAIN}\" --num_samples 102400000 --tokenizer gpt2 --seed 111 --nproc 1 --subset ${SUBSET}")
echo -n "${jid} "
done
done


SPLIT=train
for PILE_DOMAIN in "BookCorpus2" "Books3" "Gutenberg_(PG-19)" "Ubuntu_IRC" ; do
for SUBSET in 00 01 02 03 04 05 06 07 08 09 10; do
LOGDIR=logs/filter_domain_logs/${SPLIT}
mkdir -p ${LOGDIR}
jid=$(sbatch \
        --parsable \
        --partition ${PARTITION} \
        --mem 64G \
        -c 1 \
        --output ${LOGDIR}/${PILE_DOMAIN}_${SUBSET} \
        scripts/run.sh "python scripts/filter_domains.py --pile_path_dir ${PILE_PATH} --output_dir ${PREPROCESSED_OUTPUT_PATH} --intermediate_dir ${INTERMEDIATE_SCRATCH_PATH} --cache_dir ${CACHE} --split ${SPLIT} --domain \"${PILE_DOMAIN}\" --num_samples 102400000 --tokenizer gpt2 --seed 111 --nproc 1 --subset ${SUBSET}")
echo -n "${jid} "
done
done


SPLIT=validation
for PILE_DOMAIN in "ArXiv" "BookCorpus2" "Books3" "DM_Mathematics" "Enron_Emails" "EuroParl" "FreeLaw" "Github" "Gutenberg_(PG-19)" "HackerNews" "NIH_ExPorter" "OpenSubtitles" "OpenWebText2" "PhilPapers" "Pile-CC" "PubMed_Abstracts" "PubMed_Central" "StackExchange" "USPTO_Backgrounds" "Ubuntu_IRC" "Wikipedia_(en)" "YoutubeSubtitles"; do
LOGDIR=logs/filter_domain_logs/${SPLIT}
mkdir -p ${LOGDIR}
jid=$(sbatch \
        --parsable \
        --partition ${PARTITION} \
        --mem 64G \
        -c 1 \
        --output ${LOGDIR}/${PILE_DOMAIN}_${SUBSET} \
        scripts/run.sh "python scripts/filter_domains.py --pile_path_dir ${PILE_PATH} --output_dir ${PREPROCESSED_OUTPUT_PATH} --intermediate_dir ${INTERMEDIATE_SCRATCH_PATH} --cache_dir ${CACHE} --split ${SPLIT} --domain \"${PILE_DOMAIN}\" --num_samples 102400000 --tokenizer gpt2 --seed 111 --nproc 1 --subset ${SUBSET}")
echo -n "${jid} "
done
