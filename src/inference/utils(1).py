from skimage.draw import rectangle_perimeter
import skimage.io as io
from skimage.transform import resize
import numpy as np
import skimage
import pickle

import torch

from mmcv import Config, DictAction
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
import mmcv

from mmdet.datasets.pipelines import Compose  # TO LOOK AT
from mmcv.parallel import collate, scatter

from mmdet.core import bbox2result
from skimage import data, io, filters
from matplotlib.pyplot import figure

import os


class_to_number = {"Yeast White": 0, "Budding White": 1, "Yeast Opaque": 2,
                           "Budding Opaque":3,"Yeast Gray": 4, "Budding Gray": 5,
                            "Shmoo":6,"Artifact": 7, "Unknown ": 8,
                           "Pseudohyphae": 9, "Hyphae": 10, "H-junction": 11,
                           "P-junction":12,"P-Start":13,"H-Start":14}
number_to_class = {y:x for x,y in class_to_number.items()}

class Evaluator():
    def __init__(self,config,checkpoint_file):
        self.cfg = Config.fromfile(config)
        self.cfg["gpu-ids"] = 6
        self.model = build_detector(
        self.cfg.model, train_cfg=self.cfg.train_cfg, test_cfg=self.cfg.test_cfg)
        checkpoint_dict = load_checkpoint(model,checkpoint_file)
        state_dict = checkpoint_dict["state_dict"]
        self.model.CLASSES = checkpoint_dict['meta']['CLASSES']
        self.model.load_state_dict(state_dict)
        self.model.eval()
       
        
    def draw_dts(self,fname,threshold,show=True):
        data = dict(img_info=dict(filename=fname), img_prefix=None)
        test_pipeline = Compose(self.cfg.data.val.pipeline)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        x = model.extract_feat(data["img"][0]) 
        outs = self.model.bbox_head(x)  # What does this output look like? Is it per pixel? (Note: review the FCOS paper)
        bbox_list = self.model.bbox_head.get_bboxes(
            *outs, data["img_metas"][0].data[0], rescale=False)
        
        new_image = mmcv.imshow_det_bboxes(
            fname,
            bbox_list[0][0].numpy(),
            bbox_list[0][1].numpy(),
            class_names=self.model.CLASSES,
            score_thr=threshold,
            wait_time=0,
            show=False,
            out_file=None)
        figure(num=None, figsize=(20, 20))
        if show:
            io.imshow(new_image)
        else:
            return new_image
        
        
  
        
        
    def get_dts(self,filename,cls):
        cls = class_to_number[cls]
        data = dict(img_info=dict(filename=filename), img_prefix=None)
        test_pipeline = Compose(self.cfg.data.val.pipeline)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)

        x = self.model.extract_feat(data["img"][0])  # Feature maps

        outs = self.model.bbox_head(x)  # What does this output look like? Is it per pixel? (Note: review the FCOS paper)
        bbox_list = model.bbox_head.get_bboxes(
                    *outs, data["img_metas"][0].data[0], rescale=False)

        return np.array([bbox_list[0][0][i].numpy() for i in range(len(bbox_list[0][0].numpy())) if bbox_list[0][1][i] == cls])





def crop_and_resize(im,bbs):
    """
    Description: Crops the DIC image and resizes the bounding boxes
    """
    image = np.copy(im)
        
    # Identify which half of the image the bounding box is on.  Crop this half and resize.
    if bbs[0][1] > int(image.shape[1]/2):
        bbs[:,1] = bbs[:,1] - int(image.shape[1]/2)
        
        image = image[:,int(image.shape[1]/2):]
        height_scale = 800/image.shape[0]
        width_scale = 800/image.shape[1]
        
        bbs[:,[0,2]] = bbs[:,[0,2]]*height_scale
        bbs[:,[1,3]] = bbs[:,[1,3]]*width_scale
        
        image = resize(image,(800,800))
        
        return bbs, image
    
    else:
       
        image = image[:,0:int(image.shape[1]/2)]
        height_scale = 800/image.shape[0]
        width_scale = 800/image.shape[1]
        bbs[:,[0,2]] = bbs[:,[0,2]]*height_scale
        bbs[:,[1,3]] = bbs[:,[1,3]]*width_scale
        image = resize(image,(800,800))
         
        return bbs, image
    
  
def draw_bounding_boxes(image,bbs):
    """
    Description: Draws the bounding boxes and shows the image.
    """
    #y1, x1, height, width  
    tmp_image = np.copy(image)
    for i in bbs:
        rr, cc = rectangle_perimeter((i[1],i[0]), end=(i[3],i[2]),shape=tmp_image.shape,clip=True) # TEST
        tmp_image[rr,cc] = 1

    figure(num=None, figsize=(20, 20))
    io.imshow(tmp_image)
    
   
def centre_to_rect(bbs):
    # (x1,y1,width,height) -> (x1,y1,x2,y2)
    bbs[:,2] = bbs[:,0] + bbs[:,2]
    bbs[:,3] = bbs[:,1] + bbs[:,3]
    return bbs
    

def rect_to_centre(bbs):
    # (x1,y1,x2,y2) -> (x1,y1,width,height)
    bbs = np.array([np.array(i) for i in bbs])
    bbs[:,2] = bbs[:,2] - bbs[:,0]
    bbs[:,3] = bbs[:,3]