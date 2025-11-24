

model = dict(
    type='YOLOv6e',
    pretrained='weights/yolov6s.pt',  # ⭐⭐⭐ REQUIRED for 50 epoch target
    depth_multiple=0.33,
    width_multiple=0.50,
    backbone=dict(
        type='EfficientRep',
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        fuse_P2=True,
        cspsppf=True,
    ),
    neck=dict(
        type='RepBiFPANNeck',
        num_repeats=[12, 12, 12, 12],
        out_channels=[256, 128, 128, 256, 256, 512],
    ),
    head=dict(
        type='EffiDeHead',
        in_channels=[128, 256, 512],
        num_layers=3,
        begin_indices=24,
        anchors=3,
        num_classes=6,
        # ⭐ Optimized anchors (calculate from your dataset if possible)
        anchors_init=[[10,13, 16,30, 33,23],
                      [30,61, 62,45, 59,119],
                      [116,90, 156,198, 373,326]],
        out_indices=[17, 20, 23],
        strides=[8, 16, 32],
        atss_warmup_epoch=4,
        iou_type='diou',  # DIoU loss
        use_dfl=False,  
        reg_max=0,  
        distill_weight={
            'class': 1.0,
            'dfl': 1.0,
        },
    )
)

# ⭐⭐⭐ AGGRESSIVE SOLVER FOR FAST CONVERGENCE
solver = dict(
    optim='AdamW',  # Better than SGD for fast convergence
    lr_scheduler='Cosine',
    lr0=0.003,  
    lrf=0.005,  
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,  
    warmup_momentum=0.8,
    warmup_bias_lr=0.1
)

# x
data_aug = dict(
    hsv_h=0.02,  # ⭐ Increased (was 0.015)
    hsv_s=0.8,  # ⭐ Increased (was 0.7)
    hsv_v=0.5,  # ⭐ Increased (was 0.4)
    degrees=15.0,  # ⭐ More rotation (was 10.0)
    translate=0.25,  # ⭐ More translation (was 0.2)
    scale=0.95,  # ⭐ More scale (was 0.9)
    shear=3.0,  # ⭐ More shear (was 2.0)
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.2,  # ⭐ Increased (was 0.15)
    copy_paste=0.15,  # ⭐ Increased (was 0.1)
)
