from skimage.draw import rectangle_perimeter
import skimage.io as io
from skimage.transform import resize
import numpy as np
import skimage
import pickle

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
  

def save_annotations_curriculum(train_images,split,path,extra_images):
    """Save the train/val data in the COCO data format"""
    train_index = []
    val_index = []
    
    numbers = list(range(len(train_images)))
    split = int(len(train_images)*split)
    print(split)
    for k in range(split):
        i = numbers.pop(np.random.randint(len(numbers)))
        train_index.append(i)
    for k in range(split,len(train_images)):
        i = numbers.pop(np.random.randint(len(numbers)))
        val_index.append(i)
        
    len_before = len(train_images)-1  
    train_images += extra_images
    
    for i in range(len(train_images)-1,len_before,-1):
        train_index.append(i)
        val_index.append(i)
    # Save training data 
    train_data_path = path + "/train"
    train_annotations = []
    for i in train_index:
        new_entry = {}
        new_entry['filename'] = path + "/train/" + train_images[i]["filename"] 
        #new_entry['filename'] = path + "/_" + str(i) + ".png" 
        new_entry["width"] = 800
        new_entry["height"] = 800
        new_entry["ann"] = {}
        new_entry["ann"]["bboxes"] = np.array(np.array(train_images[i]["bbox"]),dtype = np.float32)
        #new_entry["ann"]["bboxes"] = np.array(train_images[i]["bbox"],dtype = np.float32)
        new_entry["ann"]["labels"] = np.array(train_images[i]["labels"],dtype = np.int64)
        train_annotations.append(new_entry)
    with open(path + "/annotations/train/" + "train_annotations.pkl" ,'wb') as f:
        pickle.dump(train_annotations, f)
        
    for i in train_index:
        image = train_images[i]["image"]
        io.imsave(path + "/train/" + train_images[i]['filename'],image)
        
        
    #Save validation data
    val_data_path = path + "/val"
    val_annotations = []
    for i in val_index:
        new_entry = {}
        new_entry['filename'] = path + "/val/" + train_images[i]["filename"] 
        #new_entry['filename'] = path + "/_" + str(i) + ".png" 
        new_entry["width"] = 800
        new_entry["height"] = 800
        new_entry["ann"] = {}
        new_entry["ann"]["bboxes"] = np.array(np.array(train_images[i]["bbox"]),dtype = np.float32)
        #new_entry["ann"]["bboxes"] = np.array(train_images[i]["bbox"],dtype = np.float32)
        new_entry["ann"]["labels"] = np.array(train_images[i]["labels"],dtype = np.int64)
        val_annotations.append(new_entry)
    with open(path + "/annotations/val/" + "val_annotations.pkl" ,'wb') as f:
        pickle.dump(val_annotations, f)
        
    for i in val_index:
        image = train_images[i]["image"]
        io.imsave(path + "/val/" + train_images[i]['filename'],image)
    
    









    
 
def save_annotations(train_images,split,path):
    """Save the train/val data in the COCO data format"""
    train_index = []
    val_index = []
    numbers = list(range(len(train_images)))
    split = int(len(train_images)*split)
    print(split)
    for k in range(split):
        i = numbers.pop(np.random.randint(len(numbers)))
        train_index.append(i)
    for k in range(split,len(train_images)):
        i = numbers.pop(np.random.randint(len(numbers)))
        val_index.append(i)
        
    # Save training data 
    train_data_path = path + "/train"
    train_annotations = []
    for i in train_index:
        new_entry = {}
        new_entry['filename'] = path + "/train/" + train_images[i]["filename"] 
        #new_entry['filename'] = path + "/_" + str(i) + ".png" 
        new_entry["width"] = 800
        new_entry["height"] = 800
        new_entry["ann"] = {}
        new_entry["ann"]["bboxes"] = np.array(np.array(train_images[i]["bbox"]),dtype = np.float32)
        #new_entry["ann"]["bboxes"] = np.array(train_images[i]["bbox"],dtype = np.float32)
        new_entry["ann"]["labels"] = np.array(train_images[i]["labels"],dtype = np.int64)
        train_annotations.append(new_entry)
    with open(path + "/annotations/train/" + "train_annotations.pkl" ,'wb') as f:
        pickle.dump(train_annotations, f)
        
    for i in train_index:
        image = train_images[i]["image"]
        io.imsave(path + "/train/" + train_images[i]['filename'],image)
        
        
    #Save validation data
    val_data_path = path + "/val"
    val_annotations = []
    for i in val_index:
        new_entry = {}
        new_entry['filename'] = path + "/val/" + train_images[i]["filename"] 
        #new_entry['filename'] = path + "/_" + str(i) + ".png" 
        new_entry["width"] = 800
        new_entry["height"] = 800
        new_entry["ann"] = {}
        new_entry["ann"]["bboxes"] = np.array(np.array(train_images[i]["bbox"]),dtype = np.float32)
        #new_entry["ann"]["bboxes"] = np.array(train_images[i]["bbox"],dtype = np.float32)
        new_entry["ann"]["labels"] = np.array(train_images[i]["labels"],dtype = np.int64)
        val_annotations.append(new_entry)
    with open(path + "/annotations/val/" + "val_annotations.pkl" ,'wb') as f:
        pickle.dump(val_annotations, f)
        
    for i in val_index:
        image = train_images[i]["image"]
        io.imsave(path + "/val/" + train_images[i]['filename'],image)
    
    