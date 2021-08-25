'''
Contains functions necessary for saving training results and generator weights.

Author: Skyler Norgaard
'''

import os

#FUNCTION FOR SAVING GENERATOR (SAVE NAME INCLUDES EPOCH FOR REPRODUCIBILITY)
def save_G(model, epoch, class_label, save_directory):
    fname = "G_epoch" + str(epoch) + "_label_class" + str(class_label) + ".h5"
    fpath = os.path.join(save_directory,fname)

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    model.save(fpath)

#FUNCTION FOR WRITING TRAINING RESULTS
def write_results(results_name, epoch, class_label, Disc_acc, GenDisc_acc, GenClass_acc,mean_RTS_sim, mean_STS_sim):
    fname = results_name

    header = "Epoch,Disc_acc,GenDisc_acc,GenClass_acc,mean_RTS_sim,mean_STS_sim" + "\n"
    to_write = str(epoch) + "," + str(Disc_acc) + "," + str(GenDisc_acc) + "," + str(GenClass_acc) + "," + str(mean_RTS_sim) + "," + str(mean_STS_sim) + "\n"
    with open(fname, "a") as f:
        if epoch==1: #this helps to separate multiple results if the code is run multiple times
            f.write(header)
            f.write(to_write)
        else:
            f.write(to_write)

