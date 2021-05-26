from utils import *
import pandas as pd
import numpy as np
import skimage
import pickle



# TODO add a config file. For now you have to hardcode the changes

ANNOTATION_PATH = "/home/data/refined/deep-microscopy/raw_annotations/varasana.json"
DATA_PATH = "/home/data/raw/deep-microscopy/Calbicans/set1/"

class_dictionary = {"Yeast White": 0, "Budding White": 1, "Yeast Opaque": 2,
                           "Budding Opaque":3,"Yeast Gray": 4, "Budding Gray": 5,
                            "Shmoo":6,"Artifact": 7, "Unknown ": 8,
                           "Pseudohyphae": 9, "Hyphae": 10, "H-junction": 11,
                           "P-junction":12,"P-Start":13,"H-Start":14}

number_to_class = {y:x for x,y in class_dictionary.items()}



ls = pd.read_excel("/home/data/refined/deep-microscopy/learning_set_feb14th.xlsx")

 
labels = pd.read_json(ANNOTATION_PATH)

labels.loc[labels['Dataset Name'].isin(['Set 2', 'Set 3', "Unlabelled Pure", "Impure Pseudohyphae"])] 



train_data = []
accepted_images = ls[ls["Learning set"] == "Yes"]["original_file_name"].values
# I need to consider two instances where a label may fail: the wrong annotations, or it was skipped.

for index, row in labels.iterrows():
    # Check that it wasn't skipped 
    if bool(row["Label"]): 
        #Check that it wasn't mislabeled.
        mislabels = len(set(j["title"] for j in row["Label"]["objects"]).difference(set(class_dictionary.keys())))
        if mislabels == 0:
            #Prepare to extract the image, bounding box coordinates and the class id
            image = io.imread(DATA_PATH + row["External ID"]) 
            bbox = []
            class_id = []
            for j in row["Label"]["objects"]:
                # Labelbox gives coordinates in (y,x,heigh,width)
                bbox.append([j["bbox"]['top'],j["bbox"]["left"],j["bbox"]["height"],j["bbox"]["width"]])    
                class_id.append(class_dictionary[j["title"]])

            # Crop half of the image and resize. For now we just select the DIC side.
            bbox, image = crop_and_resize(image,np.array(bbox))
             #This changes the bounding box coordinates from (y1,x1,height,width) to (x1,y1,x2,y2) which mmdetection requires
            bbox = np.array([np.array([i[1],i[0],i[3],i[2]]) for i in bbox]) 
            bbox = centre_to_rect(bbox)
            if row["External ID"] in accepted_images:
                train_data.append({"image": image,"bbox": bbox,"labels": class_id,'filename':row["External ID"]})

            

#split on filamentous
filamentous_data=[]

non_filamentous_data = []

filamentous_classes = {"Hyphae","Pseudohyphae","H-junction","P-junction","P-Start","H-Start"}

for i in train_data:
    classes = set(number_to_class[j] for j in i["labels"])
    if sum(1 for i in classes if i in filamentous_classes) == 0:
        non_filamentous_data.append(i)
    else:
        filamentous_data.append(i)
    
  
save_annotations(filamentous_data,.7,"/home/data/refined/deep-microscopy/train-data/varasana-filamentous")
save_annotations(non_filamentous_data,.7,"/home/data/refined/deep-microscopy/train-data/varasana-non-filamentous")
 


            
  