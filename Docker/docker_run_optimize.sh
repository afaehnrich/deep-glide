#!/bin/sh
docker run --mount src=$(pwd)/logs,target=/root/code/deep-glide/logs,type=bind afaehnrich/deep-glide:latest \
    python3 ../rl-baselines3-zoo/train.py \
    --algo sac --env JSBSim-v6 -n 100000 -optimize --n-trials 1000 --n-jobs 4 --num-threads 4 --sampler random --pruner median
