The configs subdirectory contains a python script of the form   experiment_<experiment number>.py. 

Experiment_27.py was the best result we achieved and serves as the model for version 1.0. (I didn't generalize the paths for this experiment though like I did for experiment_0.py - they are specific to our server. Sorry, but I hope the example helps.)
    
There are several examples that were used throughout  our grid search. You should be able to set most of the metaparameter of
the MMDETECTION search via the FCOS. 

Since we use a cummulative curriculum approach, this turns out to be a series of dependent executions of MMDETECTION. 

The script saves the appropriate MMDETECTION config files in each grade. There are several such files.

You can adjust parameters such as lr, momentum, decay as global variables across all grades. You can set the number of epochs and freezing strategy on a per grade basis. The grades are cummulative but enriched as follows:

(i) White, Budding White
(ii) White, Budding White, Opaque,Budding Opaque
(iii) White, Budding White, Opauqe, Budding Opaque, Gray, Budding Gray
(iii) White,Budding White, Opaque, Budding Opaque,Gray, Budding Gray, Shmoo
(iv) White, Budding White, Opaque, Budding Opaque, Gray, Budding Gray, Shmoo, Pseudohyphae, P-start, P-junction
(v) White, Budding White, Opaque, Budding Opaque, Gray, Budding Gray, Shmoo, Pseudohyphae, P-start, P-junction, Hyphae, H-start, H-junction

Artifacts and Unknowns are present in each dataset of the curriculum. 
