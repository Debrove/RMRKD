_base_ = [
    '../../deeplabv3/deeplabv3_m-v2-d8_512x512_40k_cityscapes.py'
]
# model settings
find_unused_parameters=True
alpha_mgd=0.00002
lambda_mgd=0.75
distiller = dict(
    type='SegmentationDistiller',
    teacher_pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth',
    init_student = False,
    use_logit = True,
    num_classes = 19,
    distill_cfg = [ dict(methods=[dict(type='RMRKDLoss',
                                       name='loss_rmrkd_f3',
                                       loss_weight=1.0,
                                       distance_type='cross_sim',
                                       student_channels=96,
                                       teacher_channels=1024,
                                       tau=0.1,
                                       decoupled=True,
                                       )
                                ]
                        ),
                    dict(methods=[dict(type='FeatureLoss',
                                       name='loss_mgd_fea1',
                                       student_channels=320,
                                       teacher_channels=2048,
                                       alpha_mgd=alpha_mgd,
                                       lambda_mgd=lambda_mgd
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='DecoupledKD',
                                       name='loss_dkd_channel',
                                       tau=4.0,
                                       loss_weight1=4.0,
                                       loss_weight2=4.0,
                                       dim=1,
                                       )
                                ]
                        ),
                   ]
    )

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
)

log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False),
    ])

student_cfg = 'configs/deeplabv3/deeplabv3_m-v2-d8_512x512_40k_cityscapes.py'
teacher_cfg = 'configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes.py'
