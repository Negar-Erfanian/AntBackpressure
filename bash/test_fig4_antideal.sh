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
TV=500 ;
gtype='poisson';
datapath="../data/data_${gtype}_10"
radius=0.0
max_jobs=30


echo "Fig. 4, robustness test with mixed traffic with 1 policies on network size of 100, 1x2=2 1x4=4"
# for schemes in '5' ; do
#     for load_bursty in 1 10 ; do
#         echo "submit task schemes ${schemes}, radius ${radius}";
#         python3 ANTideal_test_mixed_schemes.py --robust_test=True --datapath=${datapath} --schemes=${schemes} --out=../out  --radius=${radius} --gtype=${gtype} --T=${T} --TV=${TV} --pburst=0.5 --sizes='100' --lb=${load_bursty} --ls=1.0 &
#         running_jobs=$(jobs -p | wc -l)
#         if [[ $running_jobs -ge $max_jobs ]]; then
#             wait -n
#         fi
#     done
# done

# running_jobs=$(jobs -p | wc -l)

# if [[ $running_jobs -ge 1 ]]; then
#     wait -n
# fi

for schemes in '5' ; do
    for load_bursty in 2 20 ; do
        echo "submit task schemes ${schemes}, radius ${radius}";
        python3 ANTideal_test_mixed_schemes.py --robust_test=True --datapath=${datapath} --schemes=${schemes} --out=../out  --radius=${radius} --gtype=${gtype} --T=${T} --TV=${TV} --pburst=0.5 --sizes='100' --lb=${load_bursty} --ls=2.0 &
        running_jobs=$(jobs -p | wc -l)
        if [[ $running_jobs -ge $max_jobs ]]; then
            wait -n
        fi
    done

    for load_bursty in 0.5 5 ; do
        echo "submit task schemes ${schemes}, radius ${radius}";
        python3 ANTideal_test_mixed_schemes.py --robust_test=True --datapath=${datapath} --schemes=${schemes} --out=../out  --radius=${radius} --gtype=${gtype} --T=${T} --TV=${TV} --pburst=0.5 --sizes='100' --lb=${load_bursty} --ls=0.5&
        running_jobs=$(jobs -p | wc -l)
        if [[ $running_jobs -ge 7 ]]; then
            wait -n
        fi
    done

done