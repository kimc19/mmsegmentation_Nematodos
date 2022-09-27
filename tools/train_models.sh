#!/usr/bin/bash

python train.py ../configs/_nematodos_/segformer/segformerb0_base_80k.py
python train.py ../configs/_nematodos_/segformer/segformerb0_pretrain_80k.py
python train.py ../configs/_nematodos_/segformer/segformerb0_A2_80k.py
