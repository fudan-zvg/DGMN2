_base_ = [
    './_base_/models/setr_naive_dgmn2.py', './_base_/datasets/cityscapes_769x769.py',
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
    ),
    decode_head=dict(
        type='VisionTransformerUpHead',
        in_channels=512,
        channels=512,
        in_index=0,
        img_size=769,
        embed_dim=512,
        num_classes=19,
        norm_cfg=norm_cfg,
        dropout_ratio=0,
        num_conv=2,
        upsampling_method='bilinear',
        conv3x3_conv1x1=False,
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513))
)

# Re-config the optimizer.
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
