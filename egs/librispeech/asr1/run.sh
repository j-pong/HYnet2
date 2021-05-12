#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
train_pseudo_set="train_860_pseudo"
valid_set="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/train_asr_lstm.yaml
lm_config=conf/tuning/train_lm_transformer2.yaml
inference_config=conf/decode_asr.yaml

# --speed_perturb_factors "0.9 1.0 1.1" \
# --lm_config "${lm_config}" \
# --lm_train_text "data/${train_set}/text data/local/other_text/text" \

./asr_semi.sh \
    --audio_format flac.ark \
    --lang en \
    --ngpu 4 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --asr_config "${asr_config}" \
    --use_lm false \
    --lm_config "${lm_config}" \
    --lm_train_text "data/train_960/text data/local/other_text/text" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --train_pseudo_set "${train_pseudo_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/train_960/text" "$@"
