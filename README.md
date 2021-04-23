# deep-glide
Entwicklung einer komplexen Notlandestrategie für ein Luftfahrzeug mit Motorausfall auf Basis des Deep Reinforcement Learning.

Flugdynamisches Modell: JSBSim https://github.com/JSBSim-Team/jsbsim
Die Environments folgen dem OpenAI-Standard: https://github.com/openai/gym
Implementierung der Reinforcement-Learning-Algorithmen: https://github.com/DLR-RM/stable-baselines3
Trainiert wurde zunächst mit den Hyperparametern aus https://github.com/DLR-RM/rl-baselines3-zoo
Momentan wurden die Agenten mit SAC trainiert.

Quelle für SRTM Daten in 90m-Auflösung: https://drive.google.com/drive/folders/0B_J08t5spvd8RWRmYmtFa2puZEE und http://viewfinderpanoramas.org/Coverage%20map%20viewfinderpanoramas_org3.htm

Quelle für SRTM Daten in 30m Auflösung: https://dwtkns.com/srtm30m/

Das abstrakte Basis-Environment findet sich hier: https://github.com/afaehnrich/deep-glide/blob/master/deep_glide/jsbgym_new/abstractSimHandler.py
Hier: https://github.com/afaehnrich/deep-glide/blob/master/deep_glide/jsbgym_new/sim_handler_rl.py finden sich die daraus resultierenden, lauffähigen Environments. Die Environments unterscheiden sich vor allem durch die Reward Funktionen und die States.
Hier: https://github.com/afaehnrich/deep-glide/blob/master/deep_glide/jsbgym_new/sim_handler_2d.py findet sich die Version mit 2D-Höhendaten. Momentan noch ungetestet.
