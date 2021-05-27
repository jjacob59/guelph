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
#load_from = '/home/data/refined/deep-microscopy/output/test/output_test_pure/latest.pth'
load_from = None
resume_from = None
workflow = [('train', 1)]
