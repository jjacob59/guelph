## All these packages are not needed. To clean up.

import matplotlib.pyplot as plt

import mmcv
from mmcv.parallel import collate, scatter

import numpy as np
from skimage.draw import rectangle_perimeter
from matplotlib.pyplot import figure


import mmcv
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose


from mmcv.runner import load_checkpoint
from mmdet.apis import inference_detector, init_detector
from mmdet.models.detectors import BaseDetector


from skimage import data, io, filters
import pickle

from mmdet.datasets.pipelines import Compose
from mmdet.apis import show_result_pyplot

import sys
import glob


model_path = sys.argv[0]

val_path = sys.argv[1]

config_path = sys.argv[2]

threshold = int(sys.argv[3])


model = init_detector(config_path,model_path)


classes = {"Yeast White": 0, "Budding White": 1, "Yeast Opaque": 2,
                           "Budding Opaque":3,"Yeast Gray": 4, "Budding Gray": 5,
                            "Shmoo":6,"Pseudohyphae": 7, "Hyphae": 8, "H-junction": 9,
                           "P-junction":10,"Artifact": 11, "Unknown ": 12,"P-Start":13,"H-Start":14}
model.CLASSES = [i for i in classes]



validation_file_names = glob.glob(val_path + "/" + "*.bmp")



val_results = []

for img in validation_file_names:
    result = inference_detector(model, img)
    img = BaseDetector.show_result(img=img,
                    result=result,
                    self=model,
                    score_thr=threshold,
                    wait_time=0,
                    show=False,
                    out_file=None)

    val_results.append(img)
    
    
fig = plt.figure(figsize=(100,100))
for i in range(1,len(val_results)):
    ax1 = fig.add_subplot(15,15,i)
    ax1.imshow(val_results[i])
plt.savefig("results.png")
    
  

