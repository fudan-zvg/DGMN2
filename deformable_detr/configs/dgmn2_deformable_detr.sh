#!/usr/bin/env bash

set -x

EXP_DIR=exps/dgmn2_deformable_detr
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
