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


class Evaluator():
    def __init__(self,config,checkpoint_file,annotation_path):
        self.cfg = Config.fromfile(config)
        self.cfg["gpu-ids"] = 6
        self.model = build_detector(
        self.cfg.model, train_cfg=self.cfg.train_cfg, test_cfg=self.cfg.test_cfg)
        checkpoint_dict = load_checkpoint(model,checkpoint_file)
        state_dict = checkpoint_dict["state_dict"]
        self.model.CLASSES = checkpoint_dict['meta']['CLASSES']
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        
        self.data = []
        with (open(annotation_path, "rb")) as openfile:
            while True:
                try:
                    self.data.append(pickle.load(openfile))
                except EOFError:
                    break
        self.data = self.data[0]
        self.filenames = [i["filename"] for i in self.data]
        
        
        
    def draw_dts(self,fname,threshold):
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
        io.imshow(new_image)
        
        
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


    def get_gts(self,filename,cls):
        tmp = [i for i in self.data if i['filename'] in filename][0]
        if bool(tmp):
            bboxes = tmp["ann"]["bboxes"]
            labels = [number_to_class[i] for i in tmp["ann"]["labels"]]
            return np.array([bboxes[i] for i in range(len(bboxes)) if labels[i] == cls])
        else:
            print("Class " + cls + " not found.")
            
    
    def get_precision_recall(self,filename,cls):
        dts = self.get_dts(filename,cls)
        gts = self.get_gts(filename,cls)
        if len(gts) > 0:
            tp,fp = tpfp_default(dts,gts)
            recall = sum(tp[0])/len(gts)
            precision = sum(tp[0])/(sum(tp[0])+sum(fp[0]))
            return recall, precision
        
        
    def get_confusion(self,filename,detected_class,background_class):
        dts = self.get_dts(filename,detected_class)
        gts = self.get_gts(filename,background_class)
        if len(gts) > 0:
            tp,fp = tpfp_default(dts,gts)
            tps = sum(tp[0])
            fps = sum(fps[0])
            fns = 
            
            recall = sum(tp[0])/len(gts)
            precision = sum(tp[0])/(sum(tp[0])+sum(fp[0]))
            return recall, precision



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