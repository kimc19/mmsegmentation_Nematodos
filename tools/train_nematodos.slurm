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


WHO=`whoami`
export PATH=/home/kcarvajal/.conda/envs/openmmlab/bin:$PATH
eval "$(conda shell.bash hook)"

conda activate openmmlab

cd /work/kcarvajal/mmsegmentation_Nematodos/tools


wandb login 719a7e0be50f53f1789f774ff1535ad9e7918ba2

echo -e "\n" Modelo segformer b4 base "\n"
CUDA_VISIBLE_DEVICES=0 python train.py ../configs/_nematodos_/segformer/segformerb4_base.py
echo -e "\n" Modelo segformer b4 preentrenado "\n"
CUDA_VISIBLE_DEVICES=0 python train.py ../configs/_nematodos_/segformer/segformerb4_pretrain.py
echo -e "\n" Modelo segformer b4 con Aumentado 2 "\n"
CUDA_VISIBLE_DEVICES=0 python train.py ../configs/_nematodos_/segformer/segformerb4_A2.py

