#!/usr/bin/bash

python train.py ../configs/_nematodos_/stdc/stdc2_base.py
python train.py ../configs/_nematodos_/stdc/stdc2_pretrain.py
python train.py ../configs/_nematodos_/pointrend/pointrend_base.py
python train.py ../configs/_nematodos_/pointrend/pointrend_pretrain.py
python train.py ../configs/_nematodos_/segformer/segformerb0_base.py
python train.py ../configs/_nematodos_/segformer/segformerb0_pretrain.py
