
# Inherit and overwrite part of the config based on this config
_base_ = './rtmdet_tiny_8xb32-300e_coco.py'

data_root = 'data/ycb/' # dataset root

train_batch_size_per_gpu = 4
train_num_workers = 2

max_epochs = 20
stage2_num_epochs = 1
base_lr = 0.00008


metainfo = {
    'classes': (
    'chips_can', 'master_chef_can', 'cracker_box', 
    'sugar_box', 'tomato_soup_can', 'mustard_bottle', 
    'tuna_fish_can', 'pudding_box', 'gelatin_box', 'potted_meat_can',
    'banana', 'strawberry', 'apple', 'lemon', 'peach', 'pear', 'orange', 
    'plum', 'pitcher_base', 'bleach_cleanser', 'windex_bottle', 'wine_glass', 
    'bowl', 'mug', 'sponge', 'plate', 'fork', 'spoon', 'knife', 'spatula', 'power_drill', 
    'wood_block', 'scissors', 'padlock', 'key', 'large_marker', 'small_marker', 'adjustable_wrench',
    'phillips_screwdriver', 'flat_screwdriver', 'plastic_nut', 'hammer', 'small_clamp', 'medium_clamp', 
    'large_clamp', 'extra_large_clamp', 'mini_soccer_ball', 'softball', 'baseball', 'tennis_ball',
    'racquetball', 'golf_ball', 'chain', 'foam_brick', 'dice', 'a_marbles', 'd_marbles', 'e_marbles',
    'a_cups', 'b_cups', 'c_cups', 'd_cups', 'e_cups', 'f_cups', 'g_cups', 'h_cups', 'i_cups', 
    'j_cups', 'a_colored_wood_blocks', '1_nine_hole_peg_test', 'a_toy_airplane', 'b_toy_airplane', 
    'c_toy_airplane', 'd_toy_airplane', 'e_toy_airplane', 'f_toy_airplane', 'g_toy_airplane', 
    'h_toy_airplane', 'i_toy_airplane', 'j_toy_airplane', 'k_toy_airplane', 'a_lego_duplo', 'b_lego_duplo', 'c_lego_duplo', 'd_lego_duplo', 'e_lego_duplo', 
    'f_lego_duplo', 'g_lego_duplo', 'h_lego_duplo', 'i_lego_duplo', 'j_lego_duplo', 'k_lego_duplo', 'l_lego_duplo', 'm_lego_duplo', 'timer', 'rubiks_cube'
),
}

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='train/'),
        ann_file='train.json'))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img='val/'),
        ann_file='val.json'))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'val.json')

test_evaluator = val_evaluator

model = dict(bbox_head=dict(num_classes=96))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=10),
    dict(
        # use cosine lr from 10 to 20 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=2,  # only keep latest 2 checkpoints
        save_best='auto'
    ),
    logger=dict(type='LoggerHook', interval=5))

custom_hooks = [
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# load COCO pre-trained weight
load_from = './checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),dict(type='TensorboardVisBackend')])
