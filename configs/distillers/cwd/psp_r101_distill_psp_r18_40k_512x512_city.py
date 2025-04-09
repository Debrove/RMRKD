_base_ = [
    '../../pspnet/pspnet_r18-d8_512x512_40k_cityscapes.py'
]
# model settings
find_unused_parameters=True
distiller = dict(
    type='SegmentationDistiller',
    teacher_pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r101-d8_512x1024_40k_cityscapes/pspnet_r101-d8_512x1024_40k_cityscapes_20200604_232751-467e7cf4.pth',
    init_student = False,
    use_logit = True,
    use_adv = True,
    distill_cfg = [ dict(methods=[dict(type='CriterionAdvForG',
                                       name='loss_adv_g',
                                       adv_type='hinge',
                                       loss_weight=0.001,
                                       )
                                  ]
                         ),
                    dict(methods=[dict(type='PixelWiseLoss',
                                       name='loss_pi',
                                       loss_weight=10.0,
                                       )
                                ]
                         ),
                    ]
    )

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
)

log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False),
    ])

optimizer = dict(
    D_model=dict(type='Adam',
                  lr=0.0004,
                  betas=(0.9, 0.99)),
    model=dict(type='SGD',
                  lr=0.01,
                  momentum=0.9,
                  weight_decay=0.0005),
)

student_cfg = 'configs/pspnet/pspnet_r18-d8_512x512_40k_cityscapes.py'
teacher_cfg = 'configs/pspnet/pspnet_r101-d8_512x1024_40k_cityscapes.py'
