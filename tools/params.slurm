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

echo -e "\n" Modelo segformer b3 "\n"
python get_flops.py ../configs/_nematodos_/segformer/segformerb3_base.py
echo -e "\n" Modelo segformer b3 con aumentado 2"\n"
python get_flops.py ../configs/_nematodos_/segformer/segformerb3_A2.py
echo -e "\n" Modelo segformer b5 "\n"
python get_flops.py ../configs/_nematodos_/segformer/segformerb5_base.py
echo -e "\n" Modelo setr naive "\n"
python get_flops.py ../configs/_nematodos_/setr/SetrNaive_base.py
echo -e "\n" Modelo setr PUP "\n"
python get_flops.py ../configs/_nematodos_/setr/SetrPUP_base.py
echo -e "\n" Modelo setr MLA "\n"
python get_flops.py ../configs/_nematodos_/setr/SetrMLA_base.py
