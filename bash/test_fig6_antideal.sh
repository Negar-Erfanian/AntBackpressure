#!/bin/bash

#source .venv/bin/activate ;

# lgds = {
#     1: 'Ant-Baseline',
#     2: 'Ant-coldstart',
#     3: 'Ant-BP',
#     4: 'Ant-BP-mirror',
#     5: 'Ant-ideal',
#     6: 'SP-BP',
# }


cd ./src;

T=1000 ;
TV=1000 ;
gtype='poisson';
datapath="../data/data_${gtype}_10"
radius=0.0
max_jobs=30
sizes='100'

echo "Fig. 6, all streaming with 4 policies on 10 network sizes, 1x13=13"


for schemes in '5' ; do
    for load_streaming in 0.5 1 2 3 4 5 6 7 8 9 10 11 12 ; do
        echo "submit task schemes ${schemes}, radius ${radius}, size ${sizes}";
        python3 ANTideal_test_mixed_schemes.py --datapath=${datapath} --schemes=${schemes} --out=../out  --radius=${radius} --gtype=${gtype} --T=${T} --TV=${TV} --pburst=0.0 --sizes=${sizes} --lb=1.0 --ls=${load_streaming} &

        running_jobs=$(jobs -p | wc -l)
        if [[ $running_jobs -ge $max_jobs ]]; then
            wait -n
        fi
    done
done