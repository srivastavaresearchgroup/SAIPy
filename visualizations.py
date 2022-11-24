import matplotlib.pyplot as plt
import numpy as np

def plot_creime_data(X,y, y_pred = None):
    
    
    fig, ax = plt.subplots(4,1,figsize = [7,8])

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    p_arr = 512 - np.sum(np.array(y) > -0.5)
    
    for i in range(3):
        ax[i].plot(X[:,i], 'k')
        ax[i].set_ylim(np.min(X) * 1.05,np.max(X) * 1.05)
        ax[i].axvline(p_arr, linestyle = 'dotted', color = 'b')
    
#     ax[0].text(50, 1200, 'E component', fontsize = 13, bbox=props, family = 'serif')

        ax[3].plot(y, 'b')
        ax[3].set_ylim(-4.5,8)
        ax[3].vlines(p_arr, -4, 8, linestyle = 'dotted', color = 'b')
        ax[3].set_yticks(np.linspace(-4,6,6))
            
        if y_pred is not None:
            ax[3].plot(y_pred, 'g', label = 'CREIME Prediction')
            p_pred = 512 - np.sum(np.array(y_pred) > -0.5)
            ax[3].vlines(p_pred, -4, 8, linestyle = 'dotted', color = 'g')
            #### custom legend