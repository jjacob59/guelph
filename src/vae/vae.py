from evaluator import *
from glob import glob
import pandas as pd
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


exp = "exp27"   # this isn't so important here since we are only looking at the ground truth Varasana labels. 
# however, we need to instantiate an Evaluator object and therefore require a config.

this_experiment = "exp27_tv_objects/"

learning_set_dir = VARASANA + "train-validation/"
output = CANDESCENCE + "vaes/" 

# this points to the output of all experiments that you have run. We need this to instantiate an Evaluator
parent_dir = OUTPUT + "exp +"/"

grades = ("white","opaque","gray","shmoo","pseudohyphae","hyphae")
grade_dirs = ('white', 'white-opaque', 'white-opaque-gray', 'white-opaque-gray-shmoo', 
              'white-opaque-gray-shmoo-pseudohyphae', 'white-opaque-gray-shmoo-pseudohyphae-hyphae')

extensions = ("train_hyphae.pkl", "val_hyphae.pkl")


evals = np.full( (2), Evaluator)


for tv_idx in (0,1):
    print("\nTrain or Val: %d" %  tv_idx )
    tv = extensions[tv_idx]
    config_file = parent_dir + exp + ".py"
    if not(Path(config_file).is_file()):
        continue
    network =  parent_dir + "latest.pth"
    if not(Path(network).is_file()):
        continue
    trgt = learning_set_dir + tv
    if (tv_idx == 0):
        current_output = output + "train_object__" + exp + "/"
    else:
        current_output = output + "val_object__" + exp + "/"
    evals[tv_idx] = Evaluator(config_file, network, trgt)
    try:
        c=0
        for f in evals[tv_idx].data:
            print(f['filename'])
            bboxes = f['ann']['bboxes']
            labels = f['ann']['labels']
            if len(bboxes) > 0:
                im = mpimg.imread(f["filename"])
                for j in range(0, len(bboxes)):
                    kk = bboxes[j]
                    current_label = labels[j]
                    title = "file_" +  tv + "_" + str(c) + "_bbox_" + str(j) + "_label_" + str(current_label) + ".bmp"
                    try:
                        im1 = resize(im[int(kk[1]):int(kk[3]),int(kk[0]):int(kk[2])],(128,128,3))
                        mpimg.imsave(current_output + title, im1)
                    except:
                        print("\nIgnoring")
                c += 1
    except:
        print("\nFailed.")
        print(trgt)
        print(config_file)
        print(network)
        print(tv_idx)




