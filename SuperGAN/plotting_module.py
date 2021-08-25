'''
Contains functions for displaying and sving plots of both real and generated data (currently written for
triaxial data but could be easily modified for applications with a different number of sensor channels)

Author: Skyler Norgaard
'''


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


#FUNCTION FOR SAVING IMAGES OF PLOTTED DATA
def save_grid_plot(X,y,seq_length, sampling_rate,real, fname):
    hz = 1.0/float(sampling_rate)
    if real==True:
        title = "Real Data for label class " + str(y)
    else:
        title = "Synthetic Data for label class " + str(y)
    t = np.arange(0,seq_length*hz,hz)
    x_patch = mpatches.Patch(color='blue', label='x')
    y_patch = mpatches.Patch(color='red', label='y')
    z_patch = mpatches.Patch(color='green', label='z')
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    ax1.plot(t, x, 'b-')
    ax1.set_title(title)
    ax2.plot(t,y,'r-', label = "y")
    ax3.plot(t,z,'g-', label = "z")
    plt.legend(handles=[x_patch, y_patch, z_patch], loc=4)
    f.subplots_adjust(hspace=0)
    plt.savefig(fname)

#FUNCTION FOR DISPLAYING IMAGES OF PLOTTED DATA
def display_grid_plot(X,y, seq_length, sampling_rate, real):
    hz = 1.0/float(sampling_rate)
    if real==True:
        title = "Real Data for label class " + str(y)
    else:
        title = "Synthetic Data for label class " + str(y)
    t = np.arange(0,seq_length*hz,hz)
    x_patch = mpatches.Patch(color='blue', label='x axis')
    y_patch = mpatches.Patch(color='red', label='y axis')
    z_patch = mpatches.Patch(color='green', label='z axis')
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
    ax1.plot(t, x, 'b-')
    ax1.set_title(title)
    ax2.plot(t,y,'r-', label = "y")
    ax3.plot(t,z,'g-', label = "z")
    plt.xlabel("Seconds")
    plt.legend(handles=[x_patch, y_patch, z_patch], loc=4)
    f.subplots_adjust(hspace=0)
    plt.show()