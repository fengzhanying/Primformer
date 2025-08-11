#!/bin/bash
#SBATCH --job-name=Input
#SBATCH --partition=CU
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH --ntasks-per-node=4
#SBATCH --error=Monkey.error
#SBATCH --output=Monkey.out

rm -rf Monkey
/lustre/home/zhangfy/Pipeline/Anaconda3/envs/pytf/bin/python basenji_data.py -l 196608 -c 40960 --local -o Monkey -p 8 -t 0.1 -v 0.1 -w 128 /lustre/home/zhangfy/fetalmonkey/Macaque.t-to-t/Macaque_Assembly.v3-for-encode.fa ./Monkey_wigs.txt

rm -rf Monkey1
/lustre/home/zhangfy/Pipeline/Anaconda3/envs/pytf/bin/python basenji_data.py -l 196608 -c 40960 --local -o Monkey1 -p 24 -t 0.05 -v 0.05 -w 128 --stride_train 0.4 --stride_test 0.4 /lustre/home/zhangfy/fetalmonkey/Macaque.t-to-t/Macaque_Assembly.v3-for-encode.fa ./Monkey_wigs.txt
