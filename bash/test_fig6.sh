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
max_jobs=20
sizes='100'

echo "Fig. 6, all streaming with 4 policies on 10 network sizes, 3x10=30"
for schemes in '6' '1' '3' ; do 
    for load_streaming in '0.5' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' ; do
        echo "submit task schemes ${schemes}, radius ${radius}, size ${sizes}";
        python3 bpANT_test_mixed_schemes.py --datapath=${datapath} --schemes=${schemes} --out=../out  --radius=${radius} --gtype=${gtype} --T=${T} --TV=${TV} --pburst=0.0 --sizes=${sizes} --lb=1.0 --ls=${load_streaming} &

        running_jobs=$(jobs -p | wc -l)
        if [[ $running_jobs -ge $max_jobs ]]; then
            wait -n
        fi
    done
done
