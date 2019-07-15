#!/bin/bash
python train.py Swimmer-v1 -n 2500 -b 5
python train.py Hopper-v1 -n 30000
python train.py Walker2d-v1 -n 25000
python train.py Ant-v1 -n 100000
python train.py Reacher-v1 -n 60000 -b 50
python train.py InvertedPendulum-v1
python train.py InvertedDoublePendulum-v1 -n 12000
python train.py Humanoid-v1 -n 200000
python train.py HumanoidStandup-v1 -n 200000 -b 5