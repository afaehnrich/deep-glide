#!/bin/sh
docker run --runtime=nvidia --mount src=$(pwd)/logs,target=/root/code/deep-glide/logs,type=bind afaehnrich/deep-glide:latest python3 ../rl-baselines3-zoo/train.py --algo sac --env JSBSim-v6 -n 10000 --eval-freq 500 --save-freq 500 --tensorboard-log logs


#          volumeMounts:
#            - name: hostvol
#              mountPath: /root/code/deep-glide/logs
#      volumes:
#        - name: hostvol
#          hostPath:
#            path: /home/mixedfrog/Dokumente/kubernetes-test/logs
