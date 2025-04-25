#!/bin/bash

size=100
seed=100
# for gtype in 'ba' 'er' 'grp' 'ws'; do
#   datapath="./data/data_${gtype}_${size}"
#   echo "submit task ${gtype}";
#   python3 src/data_generation.py --datapath=${datapath} --gtype=${gtype} --size=${size} --seed=${seed};
# done

for gtype in 'poisson'; do
	datapath="./data/data_${gtype}_${size}"
	echo "submit task ${gtype}";
	python3 src/data_generation_poisson.py --datapath=${datapath} --gtype=${gtype} --size=${size} --seed=${seed};
done

echo "Done"