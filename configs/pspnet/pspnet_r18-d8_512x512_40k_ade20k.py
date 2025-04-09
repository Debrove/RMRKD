_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        num_classes=150,
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(num_classes=150, in_channels=256, channels=64))
