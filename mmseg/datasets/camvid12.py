from .builder import DATASETS
from .custom import CustomDataset

CLASSES = ('Bicyclist', 'Building', 'Car', 'Column_Pole',
           'Fence', 'Pedestrian', 'Road', 'Sidewalk',
           'SignSymbol', 'Sky', 'Tree', 'backgroud')

PALETTE = [[0, 128, 192], [128, 0, 0], [64, 0, 128], [192, 192, 128],
           [64, 64, 128], [64, 64, 0], [128, 64, 128], [0, 0, 192],
           [192, 128, 128], [128, 128, 128], [128, 128, 0], [0, 0, 0]]


@DATASETS.register_module()
class Camvid12(CustomDataset):

    CLASSES =('Bicyclist','Building','Car','Column_Pole',
              'Fence','Pedestrian','Road','Sidewalk',
              'SignSymbol','Sky','Tree','backgroud')

    PALETTE = [[0, 128, 192],[128, 0, 0],[64, 0, 128],[192, 192, 128],
               [64, 64, 128],[64, 64, 0],[128, 64, 128], [0, 0, 192],
               [192, 128, 128], [128, 128, 128],[128, 128, 0],[0,0,0]]


    def __init__(self, **kwargs):
        super(Camvid12, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_L.png',
            reduce_zero_label=False,
            classes=CLASSES,
            palette=PALETTE,
            **kwargs)

