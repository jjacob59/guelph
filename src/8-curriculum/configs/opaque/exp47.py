
_base_=[ 'config1.py' ]

optimizer = dict(lr=0.01, momentum=0.97, weight_decay=0.001, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))

load_from = '/home/data/refined/deep-microscopy/output/mike_curriculum/exp47/latest.pth'
