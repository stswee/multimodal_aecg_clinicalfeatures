#!/usr/bin/env bash

GPU=3
SESSION_NAME="multimodal_sweep"

echo "Launching sweep in tmux on GPU ${GPU}"

tmux new-session -d -s "${SESSION_NAME}" \
"export CUDA_VISIBLE_DEVICES=${GPU}; bash wave1_capacity.sh"

echo "Attach with:"
echo "tmux attach -t ${SESSION_NAME}"
