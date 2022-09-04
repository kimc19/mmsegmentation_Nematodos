from .builder import DATASETS
from .custom import CustomDataset
import os.path as osp

@DATASETS.register_module()
class NematodosDataset(CustomDataset):
    CLASSES = ('background', 'nematodo')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, split, **kwargs):
        super(NematodosDataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
