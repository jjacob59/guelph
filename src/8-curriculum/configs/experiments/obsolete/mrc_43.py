#    python ~/repo/deepmicroscopy/src/8-curriculum/mrc_40.py 

#from subprocess import Popen
import os
import sys

exp = "43"  # experiment number
gpu = "4"
lr = "0.001"
momentum = "0.97"
decay = "0.001"
order_names = ["white","opaque","gray","shmoo","pseudohyphae","hyphae"]

experiment_folder = "/home/hallett/repo/deepmicroscopy/src/8-curriculum/mike_configs/"

for i in order_names:
    experiment_file = experiment_folder + i + "/exp" + exp + ".py"
    os.system("cp " + experiment_folder + i + "/exp.py " + experiment_file )
    f = open(experiment_file, "a")
    f.write("\noptimizer = dict(lr=" + lr + ", momentum=" + momentum + ", weight_decay=" + decay + ", paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))\n")
    if (i=="white"):
        f.write("\nload_from = None\n")
    else:
        f.write("\nload_from = '/home/data/refined/deep-microscopy/output/mike_curriculum/exp" + exp +  "/latest.pth'\n")
    f.close()

for i in order_names:
    config_path = "~/repo/deepmicroscopy/src/8-curriculum/mike_configs/" + i + "/exp" + exp + ".py"
    full_command = "python /home/data/analysis-tools/mmdetection/tools/train.py" + " "\
            + config_path + " " +\
            "--work-dir=/home/data/refined/deep-microscopy/output/mike_curriculum/exp" + exp +\
            " " + "--gpu-ids=" + gpu
    os.system(full_command)
    
    
    
   

