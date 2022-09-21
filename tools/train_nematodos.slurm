#!/bin/bash
#SBATCH --job-name=nematodos
#SBATCH --output=/work/kcarvajal/mmsegmentation_Nematodos/logs/kabre.txt
#SBATCH --partition=nukwa-long
#SBATCH --ntasks=1
#SBATCH --time=168:00:00
#SBATCH --nodelist=nukwa-05.cnca
#SBATCH --mail-user=kimc19@outlook.com
#SBATCH --mail-type=ALL

module load intelpython/3.7
conda activate openmmlab

cd /work/kcarvajal/mmsegmentation_Nematodos/tools

nvidia-smi

python get_flops.py ../configs/_nematodos_/segmenter/segmenterL_base.py
python get_flops.py ./configs/_nematodos_/segmenter/segmenterL_A2.py
python get_flops.py ../configs/_nematodos_/dpt/dpt_base.py
python get_flops.py ../configs/_nematodos_/dpt/dpt_A2.py

CUDA_VISIBLE_DEVICES=0 python train.py ../configs/_nematodos_/segmenter/segmenterL_base.py
CUDA_VISIBLE_DEVICES=0 python train.py ../configs/_nematodos_/segmenter/segmenterL_pretrain.py
CUDA_VISIBLE_DEVICES=0 python train.py ../configs/_nematodos_/segmenter/segmenterL_A2.py
CUDA_VISIBLE_DEVICES=0 python train.py ../configs/_nematodos_/dpt/dpt_base.py
CUDA_VISIBLE_DEVICES=0 python train.py ../configs/_nematodos_/dpt/dpt_pretrain.py
CUDA_VISIBLE_DEVICES=0 python train.py ../configs/_nematodos_/dpt/dpt_A2.py
