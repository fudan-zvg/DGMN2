norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/dgmn2_tiny_model.pth',
    backbone=dict(
        type='dgmn2_tiny',
        norm_cfg=norm_cfg,
    ),
    decode_head=dict(
        type='VisionTransformerUpHead',
        in_channels=512,
        channels=256,
        in_index=1,
        img_size=769,
        embed_dim=512,
        num_classes=19,
        norm_cfg=norm_cfg,
        num_conv=4,
        align_corners=True,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513))
)
