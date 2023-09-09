# Sample commands to run Pile data preprocessing

# load global parameters
source constants.sh

INTERMEDIATE_SCRATCH_PATH=${CACHE}/pile_preprocessed_tmp
TOKENIZER=togethercomputer/RedPajama-INCITE-Base-7B-v0.1

SPLIT=train
for PILE_DOMAIN in "ArXiv" "DM_Mathematics" "Enron_Emails" "EuroParl" "FreeLaw" "Github" "HackerNews" "NIH_ExPorter" "OpenSubtitles" "OpenWebText2" "PhilPapers" "Pile-CC" "PubMed_Abstracts" "PubMed_Central" "StackExchange" "USPTO_Backgrounds" "Wikipedia_(en)" "YoutubeSubtitles"; do
for SUBSET in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29; do
LOGDIR=logs/preprocess_pile/${SPLIT}
mkdir -p ${LOGDIR}
jid=$(sbatch \
        --parsable \
        --partition ${PARTITION} \
        --mem 8G \
        -c 1 \
        --output ${LOGDIR}/${PILE_DOMAIN}_${SUBSET} \
        scripts/run.sh "python scripts/preprocess_pile.py --pile_path_dir ${PILE_DIR} --output_dir ${PREPROCESSED_PILE_DIR} --intermediate_dir ${INTERMEDIATE_SCRATCH_PATH} --cache_dir ${CACHE} --split ${SPLIT} --domain \"${PILE_DOMAIN}\" --tokenizer ${TOKENIZER} --seed 111 --nproc 1 --subset ${SUBSET}")
echo -n "${jid} "
done
done


SPLIT=train
for PILE_DOMAIN in "BookCorpus2" "Books3" "Gutenberg_(PG-19)" "Ubuntu_IRC" ; do
for SUBSET in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29; do
LOGDIR=logs/preprocess_pile/${SPLIT}
mkdir -p ${LOGDIR}
jid=$(sbatch \
        --parsable \
        --partition ${PARTITION} \
        --mem 64G \
        -c 1 \
        --output ${LOGDIR}/${PILE_DOMAIN}_${SUBSET} \
        scripts/run.sh "python scripts/preprocess_pile.py --pile_path_dir ${PILE_DIR} --output_dir ${PREPROCESSED_PILE_DIR} --intermediate_dir ${INTERMEDIATE_SCRATCH_PATH} --cache_dir ${CACHE} --split ${SPLIT} --domain \"${PILE_DOMAIN}\" --tokenizer ${TOKENIZER} --seed 111 --nproc 1 --subset ${SUBSET}")
echo -n "${jid} "
done
done


SPLIT=validation
for PILE_DOMAIN in "ArXiv" "BookCorpus2" "Books3" "DM_Mathematics" "Enron_Emails" "EuroParl" "FreeLaw" "Github" "Gutenberg_(PG-19)" "HackerNews" "NIH_ExPorter" "OpenSubtitles" "OpenWebText2" "PhilPapers" "Pile-CC" "PubMed_Abstracts" "PubMed_Central" "StackExchange" "USPTO_Backgrounds" "Ubuntu_IRC" "Wikipedia_(en)" "YoutubeSubtitles"; do
LOGDIR=logs/preprocess_pile/${SPLIT}
mkdir -p ${LOGDIR}
jid=$(sbatch \
        --parsable \
        --partition ${PARTITION} \
        --mem 64G \
        -c 1 \
        --output ${LOGDIR}/${PILE_DOMAIN}_${SUBSET} \
        scripts/run.sh "python scripts/preprocess_pile.py --pile_path_dir ${PILE_DIR} --output_dir ${PREPROCESSED_PILE_DIR} --intermediate_dir ${INTERMEDIATE_SCRATCH_PATH} --cache_dir ${CACHE} --split ${SPLIT} --domain \"${PILE_DOMAIN}\" --tokenizer ${TOKENIZER} --seed 111 --nproc 1 --subset ${SUBSET}")
echo -n "${jid} "
done
