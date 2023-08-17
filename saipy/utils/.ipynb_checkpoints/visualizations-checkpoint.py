import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import obspy
from obspy import UTCDateTime
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from matplotlib.dates import date2num
from matplotlib.lines import Line2D
import datetime
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import itertools
import matplotlib as mpl

def norm(X):
    maxi = np.max(abs(X), axis = 1).astype(float)
    X_ret = X.copy().astype(float)
    for i in range(X_ret.shape[0]):
        X_ret[i] = X_ret[i] / maxi[i]
    return X_ret


def plot_waveform(data, times = None, P_arr = None, S_arr = None, magnitude = None):
#     if times is None:
#         times = np.linspace(0, len(data)/100, len(data))
    fig, ax = plt.subplots(3,1,figsize = [7,6], sharex = True)
    
    for i in range(3):
        if times is None:
            ax[i].plot(data[:,i], 'k')
        else:
            ax[i].plot(times, data[:,i], 'k')
        ax[i].set_ylim(np.min(data) * 1.05,np.max(data) * 1.05)
        
        if P_arr is not None:
            ax[i].axvline(P_arr, 0.2,0.8, color = 'r', label = 'P-pick')
        if S_arr is not None:
            ax[i].axvline(S_arr, 0.2,0.8, color = 'b', label = 'S-pick')
            ax[0].legend()
        if magnitude is not None:
            fig.suptitle("Magnitude = {:.1f}".format(magnitude))
    ax[0].text(0.9, 0.85, 'E',transform=ax[0].transAxes)
    ax[1].text(0.9, 0.85, 'N',transform=ax[1].transAxes)
    ax[2].text(0.9, 0.85, 'Z',transform=ax[2].transAxes)
    
    fig.text(0.5, 0.04, 'Time (samples)', ha='center')
    fig.text(0.04, 0.5, 'Amplitude (counts)', va='center', rotation='vertical')       
    plt.show()
    return fig

def plot_creime_data(X, y, y_pred = None):
    fig, ax = plt.subplots(4,1,figsize = [7,8], sharex = True)

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    p_arr = 512 - np.sum(np.array(y) > -0.5)
    
    for i in range(3):
        ax[i].plot(X[:,i], 'k')
        ax[i].set_ylim(np.min(X) * 1.05,np.max(X) * 1.05)
        if p_arr != 512:
            ax[i].axvline(p_arr, linestyle = 'dotted', color = 'b')
    
    ax[3].plot(y, 'b')
    ax[3].set_ylim(-4.5,8)
    if p_arr != 512:
        ax[3].vlines(p_arr, -4, 8, linestyle = 'dotted', color = 'b')
    ax[3].set_yticks(np.linspace(-4,6,6))

    ax[1].set_ylabel("Amplitude (counts)", fontsize = 14)
    ax[3].set_ylabel("Data label", fontsize = 14)

    ax[3].set_xlabel("Time samples", fontsize = 14)

    if y_pred is not None:
        ax[3].plot(y_pred, 'g')
        p_pred = 512 - np.sum(np.array(y_pred) > -0.5)
        if p_pred != 512:
            ax[3].vlines(p_pred, -4, 8, linestyle = 'dotted', color = 'g')
        legend_elements = [Line2D([0], [0], color='b', label='Ground truth'),
               Line2D([0], [0], color='g', label='CREIME Prediction')]
        ax[3].legend(handles=legend_elements, loc='upper left')
        
    plt.show()
            
def plot_polarcap_data(X, y_true = None, y_pred = None):
    fig = plt.figure(figsize = [5,4])
    plt.plot(norm(np.array([X]))[0], color = 'k')
    plt.axvline(32, ls = '--', lw = 1.5, color = 'red')
    plt.xlabel('Time samples', fontsize = 14)
    plt.ylabel('Normalised\nAmplitude', fontsize = 14)
    plt.ylim(-1,1)
    
    if y_pred is not None and y_true is not None:
        plt.title('True Polarity: {}\nPredicted polarity: {}\nProbability = {:.2f}'.format(y_true[0].capitalize(), y_pred[0], y_pred[1]))
    
    elif y_pred is not None:
        plt.title('Predicted polarity: {}\nProbability = {:.2f}'.format(y_pred[0], y_pred[1]))
        
    elif y_true is not None:
        plt.title('True Polarity: {}'.format(y_true[0])) 
    plt.show()
    return fig
        
def plot_creime_rt_data(X, y, y_pred = None):
    fig, ax = plt.subplots(4,1,figsize = [7,8], sharex = True)

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    p_arr = 6000 - np.sum(np.array(y) > -0.5)
    
    for i in range(3):
        ax[i].plot(X[:,i], 'k')
        ax[i].set_ylim(np.min(X) * 1.05,np.max(X) * 1.05)
        if p_arr != 6000:
            ax[i].axvline(p_arr, linestyle = 'dotted', color = 'b')
    
#     ax[0].text(50, 1200, 'E component', fontsize = 13, bbox=props, family = 'serif')

    ax[3].plot(y, 'b')
    ax[3].set_ylim(-4.5,8)
    if p_arr != 6000:
        ax[3].vlines(p_arr, -4, 8, linestyle = 'dotted', color = 'b')
    ax[3].set_yticks(np.linspace(-4,6,6))

    ax[1].set_ylabel("Amplitude (counts)", fontsize = 14)
    ax[3].set_ylabel("Data label", fontsize = 14)

    ax[3].set_xlabel("Time samples", fontsize = 14)

    if y_pred is not None:
        ax[3].plot(y_pred, 'g')
        legend_elements = [Line2D([0], [0], color='b', label='Ground truth'),
               Line2D([0], [0], color='g', label='CREIME Prediction')]
        ax[3].legend(handles=legend_elements, loc='upper left')
    plt.show()
            
            
def convert_func(prob, n_shift):
    '''window index to sample'''
    Prob={}
    for j in range(0, len(prob)):
        id = int(j * n_shift + 200)
        Prob[id] = prob[j]
    return Prob


def plot_dynapicker_stead(stream, dataset, prob_p, prob_s, picker_num_shift, figure_size, index, normFlag=False):
    final_prob_P = convert_func(prob_p, picker_num_shift)
    final_prob_S = convert_func(prob_s, picker_num_shift)

    trace_time_p  = list(map(lambda x: x/1.0, list(final_prob_P.keys())))
    trace_time_s  = list(map(lambda x: x/1.0, list(final_prob_P.keys())))
   
    trace_data = stream[index].data
    
    if normFlag:
        trace_data = trace_data /max(np.abs(trace_data))
    
    if index == 0:
        ch = 'E-W'
    elif index == 1:
        ch = 'N-S'
    else:
        ch = 'Vertical'
        
    ## plot
    fig = plt.figure(figsize=figure_size)
    plt.suptitle(ch, y=0.93)
    axes = fig.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.1})
        
    line1, = axes[0].plot(trace_data, color='k')
    axes[0].set_ylabel('Amplitude \n (counts)')
    ymin, ymax = axes[0].get_ylim()
    pl = axes[0].vlines(dataset.attrs['p_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
    sl = axes[0].vlines(dataset.attrs['s_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
    axes[0].legend(handles=[pl, sl], loc='upper right', borderaxespad=0.)  

    axes[1].plot(trace_time_p, list(final_prob_P.values()), 'C0', label='P_prob')
    axes[1].plot(trace_time_s, list(final_prob_S.values()), 'C1', label='S_prob')
    axes[1].set_ylim(0,1)
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Probability')
    axes[1].legend(loc='upper right')
    plt.show()
    return fig
    
def plot_dynapicker_instance(stream, row, prob_p, prob_s, picker_num_shift, index, figure_size):
    
    fig, axes = plt.subplots(2, 1, figsize=figure_size, sharex=True, 
                             gridspec_kw={'hspace' : 0.08, 'height_ratios': [1, 1]}
                            )
    
    final_prob_P = convert_func(prob_p, picker_num_shift)
    final_prob_S = convert_func(prob_s, picker_num_shift)

    trace_time_p  = list(map(lambda x: x/1.0, list(final_prob_P.keys())))
    trace_time_s  = list(map(lambda x: x/1.0, list(final_prob_S.keys())))
    
    if index ==0:
        ch = 'E-W'
    elif index == 1:
        ch = 'N-S'
    else:
        ch = 'Vertical'
        
    plt.suptitle(ch, y=0.93)
    
    for j in range(3):
        if index == j:
            axes[0].plot(stream[j].data, color='k')
            
    axes[0].set_ylabel('Amplitude \n (counts)')
    ymin, ymax = axes[0].get_ylim()
    pl = axes[0].vlines(row['trace_P_arrival_sample'], ymin, ymax, color='b', linewidth=2, label='P-arrival')
    sl = axes[0].vlines(row['trace_S_arrival_sample'], ymin, ymax, color='r', linewidth=2, label='S-arrival')
    axes[0].legend(handles=[pl, sl], loc='upper right', borderaxespad=0.)  

    axes[1].plot(trace_time_p, list(final_prob_P.values()), color='C0', label='P_prob')
    axes[1].plot(trace_time_s, list(final_prob_S.values()), color='C1', label='S_prob')
    axes[1].set_ylim(0,1)
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Probability')
    axes[1].legend(loc='best')
    plt.show()
    return fig
    
def plot_dynapicker_stream(stream, prob_p, prob_s, picker_num_shift, figure_size):
    
    fig, axes = plt.subplots(2, 1, figsize=figure_size, sharex=True, 
                             gridspec_kw={'hspace' : 0.08, 'height_ratios': [1, 1]}
                            )
    
    final_prob_P = convert_func(prob_p, picker_num_shift)
    final_prob_S = convert_func(prob_s, picker_num_shift)

    trace_time_p  = list(map(lambda x: x/1.0, list(final_prob_P.keys())))
    trace_time_s  = list(map(lambda x: x/1.0, list(final_prob_S.keys())))
    
    for j in range(3):
        axes[0].plot(stream[j].data, label=stream[j].stats.channel)
        
    axes[0].set_ylabel('Amplitude \n (counts)')
    axes[0].legend(loc = 'upper right')

    axes[1].plot(trace_time_p, list(final_prob_P.values()), color='C0', label='P_prob')
    axes[1].plot(trace_time_s, list(final_prob_S.values()), color='C1', label='S_prob')
    axes[1].set_ylim(0,1)
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('Probability')
    axes[1].legend(loc='best')
    plt.show()
    
    return fig

def plot_dynapicker_train_history(train_loss, valid_loss, figure_size):
    '''
    Codes source from https://github.com/Bjarten/early-stopping-pytorch
    '''
    
    # visualize the loss as the network trained
    fig = plt.figure(figsize=figure_size)
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1 
    plt.axvline(minposs, linestyle='--', color='r', label='Early-stopping checkpoint')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, max(max(train_loss), max(valid_loss))) # consistent scale
    plt.xlim(0, len(train_loss) + 1) # consistent scale
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_dynapicker_confusionmatrix(y_true, y_pred, label_list, digits_num, figure_size, cmap=plt.cm.PuBuGn):
    '''plot comfusion matrix'''
    fig = plt.figure(figsize=figure_size)
    cm = confusion_matrix(y_true.tolist(), y_pred)
    tick_marks = np.arange(len(label_list))
    plt.xticks(tick_marks, label_list, rotation=45)
    plt.yticks(tick_marks, label_list)
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.show()
    
    ## Number of digits for formatting output floating point values.
    metrics_report = metrics.classification_report(y_true, y_pred, target_names=label_list, digits=digits_num)
    
    return fig, metrics_report

def plot_precision_recall_curve(y_true, y_pred, y_pred_prob, label_list, figure_size):
    
    y_true=np.array(y_true)
    # Compute the ROC curve for each class separately
    precisions = dict() 
    recalls = dict()
    thresholds = dict()
    roc_auc = dict()

    y_prob = np.vstack(y_pred_prob)
    n_classes = y_prob.shape[1]  # Number of classes
    
    fig = plt.figure(figsize=figure_size)
    ax= plt.axes()
    marker_list = ['o', '^']
    cmap = mpl.cm.spring
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    for i in range(n_classes-1):
        precisions[i], recalls[i], thresholds[i] = precision_recall_curve(y_true, y_prob[:,i], pos_label=2)
        plt.plot(recalls[i], precisions[i], marker=marker_list[i], linestyle='None', label=label_list[i])

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="best")
    plt.show()
    
    return fig

def plot_roc_curve(y_true, y_pred, y_pred_prob, label_list, figure_size):
    
    y_true = np.array(y_true)
    # Compute the ROC curve for each class separately
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_prob = np.vstack(y_pred_prob)
    n_classes = y_prob.shape[1]  # Number of classes

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_prob[:,i], pos_label=2)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot the ROC curves for each class
    fig = plt.figure(figsize=figure_size)
    class_label = label_list
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='Class: {0} (AUC = {1:.2f})'.format(class_label[i], roc_auc[i]))

    # plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc="best")
    plt.show()
    
    return fig
