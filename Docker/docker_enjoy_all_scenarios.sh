#!/bin/sh
N_STEPS=100000

mkdir -p logs_enjoy

# ==========================
# Tabelle 6: Basis-Szenarien
# ==========================
# Scenario A:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_A-v0 --exp-id 1 --render-mode=episode --env-kwargs render_before_reset:True
# Scenario B:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_B-v0 --exp-id 1 --render-mode=episode --env-kwargs render_before_reset:True
# Scenario C:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_C-v0 --exp-id 1 --render-mode=episode --env-kwargs render_before_reset:True

# =============================
# Tabelle 6: Szenarien mit Wind
# =============================
# Scenario C_WS:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_C_wind_konst-v0 --exp-id 1 --render-mode=episode --env-kwargs render_before_reset:True
# Scenario C_WT:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_C_wind_turb-v0 --exp-id 1 --render-mode=episode --env-kwargs render_before_reset:True
# Scenario C_WST:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_C_wind_konstturb-v0 --exp-id 1 --render-mode=episode --env-kwargs render_before_reset:True

# =============================================================
# Tabelle 6: Szenarien mit unterschiedlichen Kontrollfrequenzen
# =============================================================
# Scenario C_0.8Hz:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_C_fixed_freq-v0 --exp-id 0_8Hz --render-mode=episode --env-kwargs render_before_reset:True action_freq:0.8
mv logs_enjoy/Scenario_C_fixed_freq-v0/SAC_1 logs_enjoy/Scenario_C_fixed_freq-v0/SAC_0_8Hz
# Scenario C_0.4Hz:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_C_fixed_freq-v0 --exp-id 0_4Hz --render-mode=episode --env-kwargs render_before_reset:True action_freq:0.4
mv logs_enjoy/Scenario_C_fixed_freq-v0/SAC_1 logs_enjoy/Scenario_C_fixed_freq-v0/SAC_0_4Hz
# Scenario C_0.1Hz:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_C_fixed_freq-v0 --exp-id 0_1Hz --render-mode=episode --env-kwargs render_before_reset:True action_freq:0.1
mv logs_enjoy/Scenario_C_fixed_freq-v0/SAC_1 logs_enjoy/Scenario_C_fixed_freq-v0/SAC_0_1Hz
# Scenario C_0.05Hz:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_C_fixed_freq-v0 --exp-id 0_05Hz --render-mode=episode --env-kwargs render_before_reset:True action_freq:0.05
mv logs_enjoy/Scenario_C_fixed_freq-v0/SAC_1 logs_enjoy/Scenario_C_fixed_freq-v0/SAC_0_05Hz
# Scenario C_var0.1-2Hz:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_C_var_freq-v0 --exp-id 0_1Hz_2Hz --render-mode=episode --env-kwargs render_before_reset:True action_freq_l:0.1 action_freq_h:2.0
mv logs_enjoy/Scenario_C_var_freq-v0/SAC_1 logs_enjoy/Scenario_C_var_freq-v0/SAC_0_1Hz_2Hz
# Scenario C_var0.1-0.5Hz:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_C_var_freq-v0 --exp-id 0_1Hz_0_5Hz --render-mode=episode --env-kwargs render_before_reset:True action_freq_l:0.1 action_freq_h:0.5
mv logs_enjoy/Scenario_C_var_freq-v0/SAC_1 logs_enjoy/Scenario_C_var_freq-v0/SAC_0_1Hz_0_5Hz

# =================================================================
# Tabelle 6: Scenarien mit unterschiedl. Init-Bedingungen fürs Ziel
# =================================================================
# Scenario C_20km:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_C_different_range-v0 --exp-id 20km --render-mode=episode --env-kwargs render_before_reset:True range_rect:20000
mv logs_enjoy/Scenario_C_different_range-v0/SAC_1 logs_enjoy/Scenario_C_different_range-v0/SAC_20km
# Scenario C_40km:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_C_different_range-v0 --exp-id 40km --render-mode=episode --env-kwargs render_before_reset:True range_rect:40000
mv logs_enjoy/Scenario_C_different_range-v0/SAC_1 logs_enjoy/Scenario_C_different_range-v0/SAC_40km

# =========================================
# Tabelle 6: Basis-Szenarien mit Höhendaten
# =========================================
# Scenario A_Oz:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_A_Terrain-v0 --exp-id 1 --render-mode=episode --env-kwargs render_before_reset:True terrain:'"ocean"'
# Scenario B_Oz:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_B_Terrain-v0 --exp-id 1 --render-mode=episode --env-kwargs render_before_reset:True terrain:'"ocean"'
# Scenario C_Oz:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_C_Terrain-v0 --exp-id 1 --render-mode=episode --env-kwargs render_before_reset:True terrain:'"ocean"'

# ===================================================
# Tabelle 6: Szenarien mit verschiedenen Terraintypen
# ===================================================
# Scenario B_OzH:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_B_Terrain-v0 --exp-id OzH --render-mode=episode --env-kwargs render_before_reset:True terrain:'"oceanblock"'
mv logs_enjoy/Scenario_B_Terrain-v0/SAC_2 logs_enjoy/Scenario_B_Terrain-v0/SAC_OzH
# Scenario B_Alp:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_B_Terrain-v0 --exp-id Alp --render-mode=episode --env-kwargs render_before_reset:True terrain:'"alps"'
mv logs_enjoy/Scenario_B_Terrain-v0/SAC_2 logs_enjoy/Scenario_B_Terrain-v0/SAC_Alp
# Scenario B_AlpH:
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_B_Terrain-v0 --exp-id AlpH --render-mode=episode --env-kwargs render_before_reset:True terrain:'"block"'
mv logs_enjoy/Scenario_B_Terrain-v0/SAC_2 logs_enjoy/Scenario_B_Terrain-v0/SAC_AlpH

# ============================================
# Tabelle 7: Hat Agent das Ausweichen gelernt?
# ============================================
# Agent B_Oz, Terrain B_OzH
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_B_Terrain-v0 --exp-id 1 --render-mode=episode --env-kwargs render_before_reset:True terrain:'"oceanblock"'
mv logs_enjoy/Scenario_B_Terrain-v0/SAC_2 logs_enjoy/Scenario_B_Terrain-v0/SAC_Agent_Oz_Terrain_OzH
# Agent B_Alp, Terrain B_AlpH
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_B_Terrain-v0 --exp-id Alp --render-mode=episode --env-kwargs render_before_reset:True terrain:'"block"'
mv logs_enjoy/Scenario_B_Terrain-v0/SAC_2 logs_enjoy/Scenario_B_Terrain-v0/SAC_Agent_Alp_Terrain_AlpH
# Agent B_Oz, Terrain B_Alp
docker run --mount src=$(pwd)/logs_enjoy,target=/root/code/rl-baselines3-zoo/logs_enjoy,type=bind --mount src=$(pwd)/logs,target=/root/code/rl-baselines3-zoo/logs,type=bind afaehnrich/deep-glide:latest \
       python3 enjoy.py --algo sac -n $N_STEPS --tensorboard-log logs_enjoy --folder logs/ --env Scenario_B_Terrain-v0 --exp-id 1 --render-mode=episode --env-kwargs render_before_reset:True terrain:'"alps"'
mv logs_enjoy/Scenario_B_Terrain-v0/SAC_2 logs_enjoy/Scenario_B_Terrain-v0/SAC_Agent_Oz_Terrain_Alp

