#!/bin/sh
N_STEPS=1000000

mkdir -p logs

# ==========================
# Tabelle 6: Basis-Szenarien
# ==========================
# Scenario A:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_A-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs
# Scenario B:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_B-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs
# Scenario C:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs

# =============================
# Tabelle 6: Szenarien mit Wind
# =============================
# Scenario C_WS:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C_wind_konst-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs
# Scenario C_WT:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C_wind_turb-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs
# Scenario C_WST:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C_wind_konstturb-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs

# =============================================================
# Tabelle 6: Szenarien mit unterschiedlichen Kontrollfrequenzen
# =============================================================
# Scenario C_0.8Hz:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C_fixed_freq-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --env-kwargs action_freq:0.8
mv logs/Scenario_C_fixed_freq-v0/SAC_1 logs/Scenario_C_fixed_freq-v0/SAC_0_8Hz
mv logs/sac/Scenario_C_fixed_freq-v0_1 logs/sac/Scenario_C_fixed_freq-v0_0_8Hz
# Scenario C_0.4Hz:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C_fixed_freq-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --env-kwargs action_freq:0.4
mv logs/Scenario_C_fixed_freq-v0/SAC_1 logs/Scenario_C_fixed_freq-v0/SAC_0_4Hz
mv logs/sac/Scenario_C_fixed_freq-v0_1 logs/sac/Scenario_C_fixed_freq-v0_0_4Hz
# Scenario C_0.1Hz:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C_fixed_freq-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --env-kwargs action_freq:0.1
mv logs/Scenario_C_fixed_freq-v0/SAC_1 logs/Scenario_C_fixed_freq-v0/SAC_0_1Hz
mv logs/sac/Scenario_C_fixed_freq-v0_1 logs/sac/Scenario_C_fixed_freq-v0_0_1Hz
# Scenario C_0.05Hz:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C_fixed_freq-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --env-kwargs action_freq:0.05
mv logs/Scenario_C_fixed_freq-v0/SAC_1 logs/Scenario_C_fixed_freq-v0/SAC_0_05Hz
mv logs/sac/Scenario_C_fixed_freq-v0_1 logs/sac/Scenario_C_fixed_freq-v0_0_05Hz
# Scenario C_var0.1-2Hz:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C_var_freq-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --env-kwargs action_freq_l:0.1 action_freq_h:2.0
mv logs/Scenario_C_var_freq-v0/SAC_1 logs/Scenario_C_var_freq-v0/SAC_0_1Hz_2Hz
mv logs/sac/Scenario_C_var_freq-v0_1 logs/sac/Scenario_C_var_freq-v0_0_1Hz_2Hz
# Scenario C_var0.1-0.5Hz:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C_var_freq-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --env-kwargs action_freq_l:0.1 action_freq_h:0.5
mv logs/Scenario_C_var_freq-v0/SAC_1 logs/Scenario_C_var_freq-v0/SAC_0_1Hz_0_5Hz
mv logs/sac/Scenario_C_var_freq-v0_1 logs/sac/Scenario_C_var_freq-v0_0_1Hz_0_5Hz


# =================================================================
# Tabelle 6: Scenarien mit unterschiedl. Init-Bedingungen fürs Ziel
# =================================================================
# Scenario C_20km:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C_different_range-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --env-kwargs range_rect:20000
mv logs/Scenario_C_different_range-v0/SAC_1 logs/Scenario_C_different_range-v0/SAC_20km
mv logs/sac/Scenario_C_different_range-v0_1 logs/sac/Scenario_C_different_range-v0_20km
# Scenario C_40km:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C_different_range-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --env-kwargs range_rect:40000
mv logs/Scenario_C_different_range-v0/SAC_1 logs/Scenario_C_different_range-v0/SAC_40km
mv logs/sac/Scenario_C_different_range-v0_1 logs/sac/Scenario_C_different_range-v0_40km

# =========================================
# Tabelle 6: Basis-Szenarien mit Höhendaten
# =========================================
# Scenario A_Oz:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_A_Terrain-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --env-kwargs terrain:'"ocean"'
# Scenario B_Oz:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_B_Terrain-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --env-kwargs terrain:'"ocean"'
# Scenario C_Oz:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C_Terrain-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --env-kwargs terrain:'"ocean"'

# ===================================================
# Tabelle 6: Szenarien mit verschiedenen Terraintypen
# ===================================================
# Scenario B_OzH:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_B_Terrain-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --env-kwargs terrain:'"oceanblock"'
mv logs/Scenario_B_Terrain-v0/SAC_2 logs/Scenario_B_Terrain-v0/SAC_OzH
mv logs/sac/Scenario_B_Terrain-v0_2 logs/sac/Scenario_B_Terrain-v0_OzH
# Scenario B_Alp:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_B_Terrain-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --env-kwargs terrain:'"alps"'
mv logs/Scenario_B_Terrain-v0/SAC_2 logs/Scenario_B_Terrain-v0/SAC_Alp
mv logs/sac/Scenario_B_Terrain-v0_2 logs/sac/Scenario_B_Terrain-v0_Alp
# Scenario B_AlpH:
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_B_Terrain-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --env-kwargs terrain:'"block"'
mv logs/Scenario_B_Terrain-v0/SAC_2 logs/Scenario_B_Terrain-v0/SAC_AlpH
mv logs/sac/Scenario_B_Terrain-v0_2 logs/sac/Scenario_B_Terrain-v0_AlpH

# ==============
# Tabelle 8: SDE
# ==============
# Scenario C, kein SDE
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams use_sde:False
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_noSDE
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_noSDE
# Scenario C, SDE beim Training, alle 8 Schritte
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams sde_sample_freq:8 use_sde_at_warmup:False
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_SDE_8_nowarmup
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_SDE_8_nowarmup
# Scenario C, SDE beim Training, alle 16 Schritte
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams sde_sample_freq:16 use_sde_at_warmup:False
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_SDE_16_nowarmup
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_SDE_16_nowarmup
# Scenario C, SDE beim Training, alle 32 Schritte
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams sde_sample_freq:32 use_sde_at_warmup:False
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_SDE_32_nowarmup
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_SDE_32_nowarmup
# Scenario C, SDE beim Training, alle 64 Schritte
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams sde_sample_freq:8 use_sde_at_warmup:False
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_SDE_64_nowarmup
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_SDE_64_nowarmup
# Scenario C, SDE beim Training, 1x pro Episode
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams use_sde_at_warmup:False
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_SDE_ep_nowarmup
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_SDE_ep_nowarmup
# Scenario C, SDE bei Training und Warmup, alle 8 Schritte
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams sde_sample_freq:8 use_sde_at_warmup:True
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_SDE_8_warmup
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_SDE_8_warmup
# Scenario C, SDE bei Training und Warmup, alle 16 Schritte
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams sde_sample_freq:16 use_sde_at_warmup:True
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_SDE_16_warmup
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_SDE_16_warmup
# Scenario C, SDE bei Training und Warmup, alle 32 Schritte
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams sde_sample_freq:32 use_sde_at_warmup:True
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_SDE_32_warmup
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_SDE_32_warmup
# Scenario C, SDE bei Training und Warmup, alle 64 Schritte
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams sde_sample_freq:64 use_sde_at_warmup:True
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_SDE_64_warmup
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_SDE_64_warmup
# Scenario C, SDE bei Training und Warmup, 1x pro Episode
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams use_sde_at_warmup:True
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_SDE_ep_warmup
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_SDE_ep_warmup

# ============================
# Tabelle 9: Netzwerktopologie
# ============================
# Scenario C_Terrain, Alpen mit Hindernissen, MLP mit Layern (64, 32)
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams policy_kwargs:"dict(log_std_init=-3, net_arch=[64,32])"
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_MLP64_32
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_MLP64_32
# Scenario C_Terrain, Alpen mit Hindernissen, MLP mit Layern (256, 64)
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams policy_kwargs:"dict(log_std_init=-3, net_arch=[256,64])"
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_MLP256_64
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_MLP256_64
# Scenario C_Terrain, Alpen mit Hindernissen, MLP mit Layern (1024, 256)
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams policy_kwargs:"dict(log_std_init=-3, net_arch=[1024,256])"
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_MLP1024_256
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_MLP1024_256
# Scenario C_Terrain, Alpen mit Hindernissen, MLP mit Layern (2048, 512)
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams policy_kwargs:"dict(log_std_init=-3, net_arch=[2048,512])"
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_MLP2048_512
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_MLP2048_512
# Scenario C_Terrain, Alpen mit Hindernissen, MLP mit Layern (512, 128, 32)
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams policy_kwargs:"dict(log_std_init=-3, net_arch=[512,128,32])"
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_MLP512_128_32
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_MLP512_128_32
# Scenario C_Terrain, Alpen mit Hindernissen, MLP mit Layern (1024, 256, 64)
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams policy_kwargs:"dict(log_std_init=-3, net_arch=[1024,256,64])"
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_MLP1024_256_64
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_MLP1024_256_64
# Scenario C_Terrain, Alpen mit Hindernissen, MLP mit Layern (2048, 512, 64)
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams policy_kwargs:"dict(log_std_init=-3, net_arch=[2048,512,64])"
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_MLP2048_512_64
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_MLP2048_512_64
# Scenario C_Terrain, Alpen mit Hindernissen, MLP mit Layern (4096, 1024, 256)
docker run --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 train.py --algo sac --env Scenario_C-v0 -n $N_STEPS --eval-freq 10000 --save-freq 50000 --tensorboard-log logs --hyperparams policy_kwargs:"dict(log_std_init=-3, net_arch=[4096,1024,256])"
mv logs/Scenario_C-v0/SAC_2 logs/Scenario_C-v0/SAC_MLP4096_1024_256
mv logs/sac/Scenario_C-v0_2 logs/sac/Scenario_C-v0_MLP4096_1024_256
