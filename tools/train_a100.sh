#!bin/bash

WHO=`whoami`
export PATH=/home/$WHO/anaconda3/condabin/:$PATH
eval "$(conda shell.bash hook)"

conda activate openmmlab
cd /home/$WHO/mmsegmentation_Nematodos/tools

python train.py ../configs/_nematodos_/segformer/segformerb2_base.py
