#!/usr/bin/bash

python train.py ../configs/_nematodos_/bisenetv2/bisenetv2_base.py
python train.py ../configs/_nematodos_/bisenetv2/bisenetv2_pretrain.py
python train.py ../configs/_nematodos_/bisenetv2/bisenetv2_A2.py
