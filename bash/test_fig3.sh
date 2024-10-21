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
max_jobs=24


echo "Fig. 3, mixed traffic with 5 policies on network size of 100, 4x6=24"
for schemes in '6' '1' '3' '4' ; do 
    for load_bursty in 0.5 1 10 2 3 5 7 9 ; do
        echo "submit task schemes ${schemes}, radius ${radius}";
        python3 bpANT_test_mixed_schemes.py --datapath=${datapath} --schemes=${schemes} --out=../out  --radius=${radius} --gtype=${gtype} --T=${T} --TV=${TV} --pburst=0.5 --sizes='100' --lb=${load_bursty} --ls=1.0 &
        running_jobs=$(jobs -p | wc -l)
        if [[ $running_jobs -ge $max_jobs ]]; then
            wait -n
        fi
    done

done
