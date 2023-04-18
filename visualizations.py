import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def norm(X):
    maxi = np.max(abs(X), axis = 1)
    X_ret = X.copy()
    for i in range(X_ret.shape[0]):
        X_ret[i] = X_ret[i] / maxi[i]
    return X_ret


def plot_creime_data(X,y, y_pred = None):
    
    
    fig, ax = plt.subplots(4,1,figsize = [7,8], sharex = True)

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
        
        ax[1].set_ylabel("Amplitude (counts)", fontsize = 14)
        ax[3].set_ylabel("Data label", fontsize = 14)
        
        ax[3].set_xlabel("Time samples", fontsize = 14)
            
        if y_pred is not None:
            ax[3].plot(y_pred, 'g')
            p_pred = 512 - np.sum(np.array(y_pred) > -0.5)
            ax[3].vlines(p_pred, -4, 8, linestyle = 'dotted', color = 'g')
            legend_elements = [Line2D([0], [0], color='b', label='Ground truth'),
                   Line2D([0], [0], color='g', label='CREIME Prediction')]
            ax[3].legend(handles=legend_elements, loc='center')
            
def plot_polarcap_data(X, y_pred = None):
    plt.figure(figsize = [5,4])
    plt.plot(norm(np.array([X]))[0], color = 'k')
    plt.axvline(32, ls = '--', lw = 1.5, color = 'red')
    plt.xlabel('Time samples', fontsize = 14)
    plt.ylabel('Normalised\nAmplitude', fontsize = 14)
    plt.ylim(-1,1)
    
    if y_pred is not None:
        plt.title('Predicted polarity: {}\nProbability = {:.2f}'.format(y_pred[0], y_pred[1]))