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
max_jobs=20


echo "Fig. 4, robustness test with mixed traffic with 5 policies on network size of 100, 3x2=6 4x4=16"
# for schemes in '1' '3' '4' ; do 
#     for load_bursty in 1 10 ; do
#         echo "submit task schemes ${schemes}, radius ${radius}";
#         python3 bpANT_test_mixed_schemes.py --robust_test=True --datapath=${datapath} --schemes=${schemes} --out=../out  --radius=${radius} --gtype=${gtype} --T=${T} --TV=${TV} --pburst=0.5 --sizes='100' --lb=${load_bursty} --ls=1.0 &
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

for schemes in '6' '1' '3' '4' ; do 
    for load_bursty in 2 20 ; do
        echo "submit task schemes ${schemes}, radius ${radius}";
        python3 bpANT_test_mixed_schemes.py --robust_test=True --datapath=${datapath} --schemes=${schemes} --out=../out  --radius=${radius} --gtype=${gtype} --T=${T} --TV=${TV} --pburst=0.5 --sizes='100' --lb=${load_bursty} --ls=2.0 &
        running_jobs=$(jobs -p | wc -l)
        if [[ $running_jobs -ge $max_jobs ]]; then
            wait -n
        fi
    done

    for load_bursty in 0.5 5 ; do
        echo "submit task schemes ${schemes}, radius ${radius}";
        python3 bpANT_test_mixed_schemes.py --robust_test=True --datapath=${datapath} --schemes=${schemes} --out=../out  --radius=${radius} --gtype=${gtype} --T=${T} --TV=${TV} --pburst=0.5 --sizes='100' --lb=${load_bursty} --ls=0.5&
        running_jobs=$(jobs -p | wc -l)
        if [[ $running_jobs -ge $max_jobs ]]; then
            wait -n
        fi
    done

done
