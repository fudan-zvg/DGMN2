_base_ = [
    './_base_/models/fpn_dgmn2.py', './_base_/datasets/cityscapes_769x769.py',
    './_base_/default_runtime.py', './_base_/schedules/schedule_40k.py'
]

# Re-config the data sampler.
data = dict(samples_per_gpu=2, workers_per_gpu=2)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    pretrained='pretrained/dgmn2_small_model.pth',
    backbone=dict(
        type='dgmn2_small',
        norm_cfg=norm_cfg,
        output='neck',
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513))
)

# Re-config the optimizer.
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
