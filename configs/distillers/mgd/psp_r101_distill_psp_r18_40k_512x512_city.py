_base_ = [
    '../../pspnet/pspnet_r18-d8_512x512_40k_cityscapes.py'
]
# model settings
find_unused_parameters=True
alpha_mgd=0.00002
lambda_mgd=0.75
distiller = dict(
    type='SegmentationDistiller',
    teacher_pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x1024_40k_cityscapes/pspnet_r101-d8_512x1024_40k_cityscapes_20200604_232751-467e7cf4.pth',
    init_student = False,
    use_logit = False,
    distill_cfg = [ dict(methods=[dict(type='FeatureLoss',
                                       name='loss_mgd_fea1',
                                       student_channels=512,
                                       teacher_channels=2048,
                                       alpha_mgd=alpha_mgd,
                                       lambda_mgd=lambda_mgd
                                       )
                                  ]
                         ),
                   ]
    )

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
)

log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False),
    ])

student_cfg = 'configs/pspnet/pspnet_r18-d8_512x512_40k_cityscapes.py'
teacher_cfg = 'configs/pspnet/pspnet_r101-d8_512x1024_40k_cityscapes.py'
