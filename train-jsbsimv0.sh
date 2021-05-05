#!/bin/bash
SECONDS=0 
python3 ../rl-baselines3-zoo/train.py --algo sac --env JSBSim-v0 -n 1000000 --eval-freq 10000 --save-freq 50000 --tensorboard-log logs
echo Laufzeit: $SECONDS Sekunden

# Laufzeit 1.100 Schritte nur Simulation:
# VecEnvSubproc: 104 s
# ohne VecEnv: 101 s

# Laufzeit 1.100 Schritte Training:
# VecEnvSubproc: 125 s
# ohne VecEnv: 118 s
# 3x parallel : 118s
