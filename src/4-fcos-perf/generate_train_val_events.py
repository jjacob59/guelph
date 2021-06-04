from evaluator import *
from glob import glob

import pandas as pd
from pathlib import Path
import numpy as np

from functools import reduce
from collections import defaultdict

import json


numba = "_exp27_val_"
exp = 'exp27' 
target = "val"   # can be {val | train}
threshes = [0.1]   #  , 0.3, 0.33, 0.35, 0.4, 0.5, 0.6]

parent_dir = "/home/data/refined/deep-microscopy/output/final_experiment/"   # this points to the output of all experiments that you have run.
dataset_dir = "/home/data/refined/deep-microscopy/train-data/final/"
performance_save = "/home/data/refined/deep-microscopy/performance/" + exp + "_results/"

grades = ("white","opaque","gray","shmoo","pseudohyphae","hyphae")
grade_dirs = ('white', 'white-opaque', 'white-opaque-gray', 'white-opaque-gray-shmoo', 
              'white-opaque-gray-shmoo-pseudohyphae', 'white-opaque-gray-shmoo-pseudohyphae-hyphae')
cls = 15


evals = np.full((len(threshes), len(grades)), Evaluator)

for t in range(0, len(threshes)):
    thresh = threshes[t]
    for grade_num in range(0, len(grades)):
        print("Thresh: " + str(t) + " Grade " + str( grade_num ) )
        config_file = parent_dir + exp + "/" + exp + ".py"
        if not(Path(config_file).is_file()):
            continue
        network =  parent_dir + exp + "/latest.pth"
        if not(Path(network).is_file()):
            continue
        dataset = dataset_dir + target + "_" + str(grades[grade_num]) + ".pkl"
        try:
            evals[t, grade_num] = Evaluator(config_file, network, dataset)
            evals[t, grade_num].get_blindspots(threshold = thresh)  # sets self.blindspots
            evals[t, grade_num].get_mirages(threshold = thresh)     # sets self.mirages
            evals[t, grade_num].get_classification_errors(threshold = thresh)   # sets self.classification_errors
            evals[t, grade_num].get_classification_good(threshold = thresh)     # sets self.classification_good
        except:
            print("\nFailed on " + exp + " " + str(thresh))
            print(dataset)
            print(config_file)
            print(network)


all_events = pd.DataFrame( columns = ['event', 'filename', 'experiment', 'grade', 'threshold', 'bbox_1', 'bbox_2','bbox_3','bbox_4', 'gt_class', 'dt_class' ] ) 

for t in range(0, len(threshes)):
    thresh = threshes[t]
    for grade_num in range(0, len(grades)):
        print("Thresh: " + str(t) + " Grade " + str( grade_num ) )
        
        # blindspots
        for b in evals[t, grade_num].blindspots['all']:
            if len(b['bboxes']) > 0:
                fname = b['filename']
                clss = b['classes']
                bb = b['bboxes']['fuck']
                ttt = bb.tolist()
                bb1 = [item[0] for item in ttt]
                bb2 = [item[1] for item in ttt]
                bb3 = [item[2] for item in ttt]
                bb4 = [item[3] for item in ttt]
                
                tmp_events = pd.DataFrame.from_dict(dict( [
                                ('event', ['blindspot']*len(clss)), 
                                ('filename', [fname]*len(clss)), 
                                ('experiment', [exp]*len(clss)), 
                                ('grade', [grades[grade_num]]*len(clss)), 
                                ('threshold', [thresh]*len(clss)),
                                ('bbox_1', bb1), 
                                ('bbox_2', bb2), 
                                ('bbox_3', bb3), 
                                ('bbox_4', bb4), 
                                ('gt_class', [evals[t, grade_num].model.CLASSES[i] for i in clss]), 
                                ('dt_class', [np.nan]*len(clss) ) ] ))
                all_events = all_events.append( tmp_events )
        
        # hallucinations (previously known as mirages)
        for b in evals[t, grade_num].mirages['all']:
            if len(b['bboxes']) > 0:
                fname = b['filename']
                clss = b['classes']
                bb = b['bboxes']['hallucination']
                ttt = bb.tolist()
                bb1 = [item[0] for item in ttt]
                bb2 = [item[1] for item in ttt]
                bb3 = [item[2] for item in ttt]
                bb4 = [item[3] for item in ttt]
                
                tmp_events = pd.DataFrame.from_dict(dict( [
                                ('event', ['hallucination']*len(clss)), 
                                ('filename', [fname]*len(clss)), 
                                ('experiment', [exp]*len(clss)), 
                                ('grade', [grades[grade_num]]*len(clss)), 
                                ('threshold', [thresh]*len(clss)),
                                ('bbox_1', bb1), 
                                ('bbox_2', bb2), 
                                ('bbox_3', bb3), 
                                ('bbox_4', bb4), 
                                ('gt_class', [np.nan]*len(clss)), 
                                ('dt_class', [evals[t, grade_num].model.CLASSES[i] for i in clss] ) ]))
                all_events = all_events.append( tmp_events )
            
        # classification errors
        for b in range(0, len(evals[t, grade_num].model.CLASSES)):
            target_cls = evals[t, grade_num].classification_errors[evals[t, grade_num].model.CLASSES[b]]
            for c in target_cls:
                if len(c['bboxes']) > 0:
                    fname = c['filename']
                    kys= [*c['bboxes']]
                    for k in kys:
                        current = c['bboxes'][k]
                        ttt = current.tolist()
                        bb1 = [item[0] for item in ttt]
                        bb2 = [item[1] for item in ttt]
                        bb3 = [item[2] for item in ttt]
                        bb4 = [item[3] for item in ttt]
 
                        tmp_events = pd.DataFrame.from_dict(dict( [
                                ('event', ['class_error']*len(current)), 
                                ('filename', [fname]*len(current)), 
                                ('experiment', [exp]*len(current)), 
                                ('grade', [grades[grade_num]]*len(current)), 
                                ('threshold', [thresh]*len(current)),
                                ('bbox_1', bb1), 
                                ('bbox_2', bb2), 
                                ('bbox_3', bb3), 
                                ('bbox_4', bb4), 
                                ('gt_class', [k]*len(current)), 
                                ('dt_class', evals[t, grade_num].model.CLASSES[b] ) ]))
                        all_events = all_events.append( tmp_events )
           
        # true positives  
        for b in range(0, len(evals[t, grade_num].model.CLASSES)):
            target_cls = evals[t, grade_num].classification_good[evals[t, grade_num].model.CLASSES[b]]
            for c in target_cls:
                if len(c['bboxes']) > 0:
                    fname = c['filename']
                    kys= [*c['bboxes']]
                    for k in kys:
                        current = c['bboxes'][k]
                        ttt = current.tolist()
                        bb1 = [item[0] for item in ttt]
                        bb2 = [item[1] for item in ttt]
                        bb3 = [item[2] for item in ttt]
                        bb4 = [item[3] for item in ttt]
                        tmp_events = pd.DataFrame.from_dict(dict( [
                                ('event', ['class_good']*len(current)), 
                                ('filename', [fname]*len(current)), 
                                ('experiment', [exp]*len(current)), 
                                ('grade', [grades[grade_num]]*len(current)), 
                                ('threshold', [thresh]*len(current)),
                                ('bbox_1', bb1), 
                                ('bbox_2', bb2), 
                                ('bbox_3', bb3), 
                                ('bbox_4', bb4), 
                                ('gt_class', [k]*len(current)), 
                                ('dt_class', evals[t, grade_num].model.CLASSES[b] ) ]))
                        all_events = all_events.append( tmp_events )
             
    # save to file so that it can be picked up in R.
    all_events.to_csv(performance_save + "all_events" + numba + "_thresh_" + str(thresh) + ".csv")
        
            
            



