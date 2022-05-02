_base_ = [
    './_base_/models/setr_mla_dgmn2.py', './_base_/datasets/cityscapes_769x769.py',
    './_base_/default_runtime.py', './_base_/schedules/schedule_40k.py'
]

# Re-config the data sampler.
data = dict(samples_per_gpu=2, workers_per_gpu=2)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='pretrained/dgmn2_large_model.pth',
    backbone=dict(
        type='dgmn2_large',
        norm_cfg=norm_cfg,
        output='neck',
    ),
    decode_head=dict(
        type='VIT_MLAHead',
        in_channels=512,
        channels=512,
        img_size=769,
        mla_channels=[64, 128, 320, 512],
        mlahead_channels=128,
        num_classes=19,
        dropout_ratio=0,
        norm_cfg=norm_cfg,
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513))
)

# Re-config the optimizer.
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
