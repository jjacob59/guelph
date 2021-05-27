_base_ = [ 'config.py' ]

optimizer = dict(lr=0.01, momentum=0.9, weight_decay=0.0001, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))

load_from = '/home/data/refined/deep-microscopy/output/mike_curriculum/exp44/latest.pth'
