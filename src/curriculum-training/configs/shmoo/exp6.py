
_base_=[ 'config2.py' ]

total_epochs = 500

optimizer = dict(lr=0.01, momentum=0.9, weight_decay=0.001, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))

load_from = '/home/data/refined/deep-microscopy/output/final_experiment/exp6/latest.pth'
