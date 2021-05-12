#!/bin/sh
docker run --mount src=$(pwd)/logs,target=/root/code/deep-glide/logs,type=bind afaehnrich/deep-glide:latest python3 ../rl-baselines3-zoo/train.py --algo sac --env JSBSim-v2 -n 1000 --eval-freq 1000 --save-freq 500 --tensorboard-log logs


#          volumeMounts:
#            - name: hostvol
#              mountPath: /root/code/deep-glide/logs
#      volumes:
#        - name: hostvol
#          hostPath:
#            path: /home/mixedfrog/Dokumente/kubernetes-test/logs

          volumeMounts:
            - name: hostvol
              mountPath: /root/code/deep-glide/logs
      volumes:
        - name: hostvol
          hostPath:
            path: /home/mixedfrog/Dokumente/kubernetes-test/logs
