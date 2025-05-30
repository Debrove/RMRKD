# RMRKD: Region-aware mutual relational knowledge distillation for semantic segmentation

## Environment
1. Clone our repo and create conda environment.
```
git clone https://github.com/Debrove/RMRKD.git && cd RMRKD
conda create -n rmrkd python=3.8
conda activate rmrkd
```

2. Install Pytorch and other dependencies
Please refer MMSegmentation for detail installation.
```
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install openmim
mim install mmcv-full==1.7.0
mim install mmsegmentation==0.29.1
pip install -r requirements.txt
```

## Dataset
Please follow MMSegmentation to prepare datasets.

## Train

```
#single GPU
python tools/train.py configs/distillers/rmrkd/psp_r101_distill_psp_r18_40k_512x512_city.py

#multi GPU
bash tools/dist_train.sh configs/distillers/rmrkd/psp_r101_distill_psp_r18_40k_512x512_city.py 8
```

## Transfer
```
# Tansfer the RMRKD model into mmseg model
python pth_transfer.py --mgd_path $ckpt --output_path $new_mmseg_ckpt
```
## Test

```
#single GPU
python tools/test.py configs/pspnet/pspnet_r18-d8_512x512_40k_cityscapes.py $new_mmseg_ckpt --eval mIoU

#multi GPU
bash tools/dist_test.sh configs/pspnet/pspnet_r18-d8_512x512_40k_cityscapes.py $new_seg_ckpt 8 --eval mIoU
```

## Results on CityScapes
|  Teacher  |  Student   | Baseline(mIoU) | +RMRKD(mIoU) |                            config                            |
| :------: | :------: | :----------------: | :------------: | :----------------------------------------------------------: |
|   PspNet-R101   | PspNet-R18 |        69.37        |      75.72      | [config](https://github.com/Debrove/RMRKD/tree/master/configs/distillers/rmrkd/psp_r101_distill_psp_r18_40k_512x512_city.py) |
| PspNet-R101 | DeepLabV3-R18 |        73.37        |      76.72      | [config](https://github.com/Debrove/RMRKD/tree/master/configs/distillers/rmrkd/psp_r101_distill_deepv3_r18_40k_512x512_city.py) |
|   DeepLabV3 plus-R101   | MobileNetV2 |        73.76        |      76.87      | [config](https://github.com/Debrove/RMRKD/tree/master/configs/distillers/rmrkd/deepv3plus_r101_distill_deepv3_mbv2_40k_512x512_city.py) |
| DeepLabV3-R101 | MobileNetV2 |        73.11        |      76.32      | [config](https://github.com/Debrove/RMRKD/tree/master/configs/distillers/rmrkd/deepv3_r101_distill_deepv3_mbv2_40k_512x512_city.py) |

## Results on Pascal VOC
|  Teacher  |  Student   | Baseline(mIoU) | +RMRKD(mIoU) |                            config                            |
| :------: | :------: | :----------------: | :------------: | :----------------------------------------------------------: |
|   PspNet-R101   | PspNet-R18 |        70.52        |      74.64      | [config](https://github.com/Debrove/RMRKD/tree/master/configs/distillers/rmrkd/psp_r101_distill_psp_r18_40k_512x512_voc12aug.py) |
| PspNet-R101 | DeepLabV3-R18 |        71.60        |      74.97      | [config](https://github.com/Debrove/RMRKD/tree/master/configs/distillers/rmrkd/psp_r101_distill_deepv3_r18_40k_512x512_voc12aug.py) |


## Acknowledgements
Our code is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [MGD](https://github.com/yzd-v/MGD.git), [CIRKD](https://github.com/winycg/CIRKD). Many thanks to these great works and open-source codebases.

## Citation
```
@article{zheng2025region,
  title={Region-aware mutual relational knowledge distillation for semantic segmentation},
  author={Zheng, Haowen and Lin, Xuxin and Liang, Hailun and Zhou, Benjia and Liang, Yanyan},
  journal={Pattern Recognition},
  volume={161},
  pages={111319},
  year={2025},
  publisher={Elsevier}

```

