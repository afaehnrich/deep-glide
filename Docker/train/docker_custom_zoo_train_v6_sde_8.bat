docker pull afaehnrich/deep-glide-custom-zoo:latest
docker run --mount src=%~dp0/../../../rl-baselines3-zoo/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind ^
			--mount src=%~dp0/../../../rl-baselines3-zoo/rl-trained-agents,target=/root/code/rl-baselines3-zoo/rl-trained-agents,type=bind ^
			--mount src=%~dp0/../../../rl-baselines3-zoo/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind ^
       afaehnrich/deep-glide-custom-zoo:latest python3 train.py --algo sac --env JSBSim-v6 -n 1000000 --eval-freq 10000 --save-freq 50000 --tensorboard-log logs ^
			--hyperparams sde_sample_freq:8 use_sde_at_warmup:True
			
