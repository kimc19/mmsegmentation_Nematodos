python test.py ../configs/nematodos/stdc/stdc1_base.py ../work_dirs/stdc1_base/latest.pth --eval mIoU --show-dir ../results/stdc1_base > ../logs/test_stdc1_base.txt 2>&1

python test.py ../configs/nematodos/stdc/stdc2_base.py ../work_dirs/stdc2_base/latest.pth --eval mIoU --show-dir ../results/stdc2_base > ../logs/test_stdc2_base.txt 2>&1

python test.py ../configs/nematodos/stdc/stdc2_pretrain.py ../work_dirs/stdc2_pretrain/latest.pth --eval mIoU --show-dir ../results/stdc2_pretrain > ../logs/test_stdc2_pretrain.txt 2>&1

python test.py ../configs/nematodos/stdc/stdc2_A2.py ../work_dirs/stdc2_A2/latest.pth --eval mIoU --show-dir ../results/stdc2_A2 > ../logs/test_stdc2_A2.txt 2>&1

python test.py ../configs/nematodos/pointrend/pointrend_base.py ../work_dirs/PointRend_base/latest.pth --eval mIoU --show-dir ../results/pointrend_base > ../logs/test_pointrend_base.txt 2>&1

python test.py ../configs/nematodos/pointrend/pointrend_pretrain.py ../work_dirs/PointRend_pretrain/latest.pth --eval mIoU --show-dir ../results/pointrend_pretrain > ../logs/test_pointrend_pretrain.txt 2>&1

python test.py ../configs/nematodos/pointrend/pointrend_A2.py ../work_dirs/PointRend_A2/latest.pth --eval mIoU --show-dir ../results/pointrend_A2 > ../logs/test_pointrend_A2.txt 2>&1

python test.py ../configs/nematodos/segformer/segformerb0_base.py ../work_dirs/segformerb0_base/latest.pth --eval mIoU --show-dir ../results/segformerb0_base > ../logs/test_segformerb0_base.txt 2>&1

python test.py ../configs/nematodos/segformer/segformerb0_base_80k.py ../work_dirs/segformerb0_base_80k/latest.pth --eval mIoU --show-dir ../results/segformerb0_base_80k > ../logs/test_segformerb0_base_80k.txt 2>&1

python test.py ../configs/nematodos/segformer/segformerb0_pretrain_80k.py ../work_dirs/segformerb0_pretrain_80k/latest.pth --eval mIoU --show-dir ../results/segformerb0_pretrain_80k > ../logs/test_segformerb0_pretrain_80k.txt 2>&1

python test.py ../configs/nematodos/segformer/segformerb0_A2_80k.py ../work_dirs/segformerb0_A2_80k/latest.pth --eval mIoU --show-dir ../results/segformerb0_A2_80k > ../logs/test_segformerb0_A2_80k.txt 2>&1


