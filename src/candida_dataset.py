from .custom import CustomDataset
from .builder import DATASETS
import json
import pickle

@DATASETS.register_module
class CandidaDataset(CustomDataset):
    
    # PURE CLASSIFIER 
    CLASSES = ("Yeast White", "Budding White", "Yeast Opaque","Budding Opaque","Yeast Gray", "Budding Gray",
               "Shmoo", "Pseudohyphae", "Hyphae", "H-junction","P-junction","Artifact", "Unknown ","P-Start","H-Start")
    
   
   # If I want this to work.. I will need to rewrite this custom dataset...
    def load_annotations(self, ann_file):
           
        self.cat2label = {"Yeast White": 0, "Budding White": 1, "Yeast Opaque": 2,
                           "Budding Opaque":3,"Yeast Gray": 4, "Budding Gray": 5,
                            "Shmoo":6,"Artifact": 7, "Unknown ": 8,
                           "Pseudohyphae": 9, "Hyphae": 10, "H-junction": 11,
                           "P-junction":12,"P-Start":13,"H-Start":14}
       
        data = []
        with (open(ann_file, "rb")) as openfile:
            while True:
                try:
                    data.append(pickle.load(openfile))
                except EOFError:
                    break  
        return data[0]

    def get_ann_info(self, idx):
        return self.data_infos[idx]["ann"]
    
    
  


@DATASETS.register_module
class FilamentousDataset(CustomDataset):
        
    # PURE CLASSIFIER 
    CLASSES = ("Yeast White", "Budding White", "Yeast Opaque","Budding Opaque","Yeast Gray", "Budding Gray",
               "Shmoo","Artifact", "Unknown ","Pseudohyphae", "Hyphae", "H-junction","P-junction","P-Start","H-Start")
    
   
   # If I want this to work.. I will need to rewrite this custom dataset...
    def load_annotations(self, ann_file):
           
        self.cat2label = {"Yeast White": 0, "Budding White": 1, "Yeast Opaque": 2,
                           "Budding Opaque":3,"Yeast Gray": 4, "Budding Gray": 5,
                            "Shmoo":6,"Artifact": 7, "Unknown ": 8,
                           "Pseudohyphae": 9, "Hyphae": 10, "H-junction": 11,
                           "P-junction":12,"P-Start":13,"H-Start":14}
       
        data = []
        with (open(ann_file, "rb")) as openfile:
            while True:
                try:
                    data.append(pickle.load(openfile))
                except EOFError:
                    break  
        return data[0]

    def get_ann_info(self, idx):
        return self.data_infos[idx]["ann"]
    
    
 
@DATASETS.register_module
class CurriculumDataset(CustomDataset):
        
    # PURE CLASSIFIER 
    CLASSES = ("Yeast White", "Budding White", "Yeast Opaque","Budding Opaque","Yeast Gray", "Budding Gray",
               "Shmoo","Artifact", "Unknown ","Pseudohyphae", "Hyphae", "H-junction","P-junction","P-Start","H-Start")
    
   
   # If I want this to work.. I will need to rewrite this custom dataset...
    def load_annotations(self, ann_file):
           
        self.cat2label = {"Yeast White": 0, "Budding White": 1, "Yeast Opaque": 2,
                           "Budding Opaque":3,"Yeast Gray": 4, "Budding Gray": 5,
                            "Shmoo":6,"Artifact": 7, "Unknown ": 8,
                           "Pseudohyphae": 9, "Hyphae": 10, "H-junction": 11,
                           "P-junction":12,"P-Start":13,"H-Start":14}
       
        data = []
        with (open(ann_file, "rb")) as openfile:
            while True:
                try:
                    data.append(pickle.load(openfile))
                except EOFError:
                    break  
        return data[0]

    def get_ann_info(self, idx):
        return self.data_infos[idx]["ann"]
    
  

@DATASETS.register_module
class NonFilamentousDataset(CustomDataset):
        
    # PURE CLASSIFIER 
    CLASSES = ("Yeast White", "Budding White", "Yeast Opaque","Budding Opaque","Yeast Gray", "Budding Gray",
               "Shmoo","Artifact","Unknown")
    
   
   # If I want this to work.. I will need to rewrite this custom dataset...
    def load_annotations(self, ann_file):
           
        self.cat2label =  self.cat2label = {"Yeast White": 0, "Budding White": 1, "Yeast Opaque": 2,
                           "Budding Opaque":3,"Yeast Gray": 4, "Budding Gray": 5,
                            "Shmoo":6,"Artifact": 7, "Unknown ": 8}
       
        data = []
        with (open(ann_file, "rb")) as openfile:
            while True:
                try:
                    data.append(pickle.load(openfile))
                except EOFError:
                    break  
        return data[0]

    def get_ann_info(self, idx):
        return self.data_infos[idx]["ann"]