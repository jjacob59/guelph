#    python ~/repo/deepmicroscopy/src/8-curriculum/mrc_40.py 
#from subprocess import Popen
import os
import sys

resume = False # flip to true if you want to restart after a crash
current = "white"

exp = "65"  # experiment number
gpu = "5"
lr = "0.001"
momentum = "0.9"
decay = "0.0001"
total_epochs={"white":10000,"opaque":10000,"gray":10000,"shmoo":10000,"pseudohyphae":10000  ,"hyphae":10000}


grades = {"white":1, "opaque":2,"gray":3,"shmoo":4,"pseudohyphae":5  ,"hyphae":6}
total_epochs={"white":10000,"opaque":10000,"gray":10000,"shmoo":10000,"pseudohyphae":10000  ,"hyphae":10000}

order_names = ["white","opaque","gray","shmoo","pseudohyphae","hyphae"]
freeze = {"white":1,"opaque":1,"gray":2,"shmoo":2,"pseudohyphae":2  ,"hyphae":2}

experiment_folder = "/home/hallett/repo/deepmicroscopy/src/8-curriculum/mike_configs/"

if (resume==True):
    order_names_p = order_names[grades[current]-1:len(order_names)]       
        
    for j in order_names_p:
        config_path = "~/repo/deepmicroscopy/src/8-curriculum/mike_configs/" + j + "/exp" + exp + ".py"
        output_folder = "/home/data/refined/deep-microscopy/output/mike_curriculum/exp" + exp 
        full_command = "python /home/data/analysis-tools/mmdetection/tools/train.py" + " "\
            + config_path + " " +\
            "--work-dir=" + output_folder 
        full_command = full_command + " --resume-from=/home/data/refined/deep-microscopy/output/mike_curriculum/exp" + exp + "/latest.pth" 
        full_command = full_command + " " + "--gpu-ids=" + gpu
        try:
            os.system(full_command)
            try:
                os.mkdir(output_folder + "/" + j)
            except OSError:
                print ("Creation of the directory %s failed" % j)
            os.system("cp "+ output_folder + "/*" + " " +  output_folder + "/" + j)
        except:
            print("\n\nSystem failed with grade ", j)
      
else:

    for i in order_names:
        experiment_file = experiment_folder + i + "/exp" + exp + ".py"
        # os.system("cp " + experiment_folder + i + "/exp.py " + experiment_file )
        f = open(experiment_file, "w")
        f.write("\n_base_=[ 'config" + str(freeze[i]) + ".py' ]\n")
        f.write("\ntotal_epochs = " + str(total_epochs[i]) + "\n")
        f.write("\noptimizer = dict(lr=" + lr + ", momentum=" + momentum + ", weight_decay=" + decay + ", paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))\n")
        if (i=="white"):
            f.write("\nload_from = None\n")
        else:
            f.write("\nload_from = '/home/data/refined/deep-microscopy/output/mike_curriculum/exp" + exp +  "/latest.pth'\n")
        f.close()

  

    for i in order_names:
        config_path = "~/repo/deepmicroscopy/src/8-curriculum/mike_configs/" + i + "/exp" + exp + ".py"
        output_folder = "/home/data/refined/deep-microscopy/output/mike_curriculum/exp" + exp 
        full_command = "python /home/data/analysis-tools/mmdetection/tools/train.py" + " "\
            + config_path + " " +\
            "--work-dir=" + output_folder
        full_command = full_command + " " + "--gpu-ids=" + gpu
        try:
            os.system(full_command)
            try:
                os.mkdir(output_folder + "/" + i)
            except OSError:
                print ("Creation of the directory %s failed" % i)
            os.system("cp "+ output_folder + "/*" + " " +  output_folder + "/" + i)
        except:
            print("\n\nSystem failed with grade ", i)
             
   
os.system("curl -X POST  https://maker.ifttt.com/trigger/{finished}/with/key/nLhA1COLNOzlGa0dOawFP7sO3U_IfHzoGf4Z7ajHjgo?value1=" + exp)

