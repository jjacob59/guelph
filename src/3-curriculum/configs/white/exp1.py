
_base_=[ 'config1.py' ]

total_epochs = 500

optimizer = dict(lr=0.001, momentum=0.9, weight_decay=0.0001, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))

load_from = None
