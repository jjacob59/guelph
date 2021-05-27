checkpoint_config = dict(interval=300)
# yapf:disable
log_config = dict(
    interval=15,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
#load_from = '/home/data/refined/deep-microscopy/output/mike_curriculum/exp40/latest.pth'
resume_from = None
workflow = [('train', 1)]
